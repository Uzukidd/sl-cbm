import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import cv2
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

from utils import CBM_Net
from typing import Callable
import pdb


def averageDropIncrease(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    masked_images: torch.Tensor,
):
    """
    Args:
        model: nn.Module
        images: [B, 3, H, W]
        masked_images: [B, 3, H, W]
        labels: [B, ]

    """
    images = images
    masked_images = masked_images

    logits = model(images)
    outputs = torch.nn.functional.softmax(logits, dim=1)  # [B, C]
    if labels == None:
        predict_labels = outputs.argmax(axis=1)  # [B, ]
    else:
        labels = labels  # [B, ]
        predict_labels = labels  # [B, ]

    logits_mask = model(masked_images)
    outputs_mask = torch.nn.functional.softmax(logits_mask, dim=1)  # [B, C]

    one_hot_labels = torch.eye(outputs.size(1)).to(images.device)[
        predict_labels
    ]  # [B, C]
    Y = torch.masked_select(outputs, one_hot_labels.bool())  # [B, C] -> [B, ]

    one_hot_labels_mask = torch.eye(outputs.size(1)).to(images.device)[
        predict_labels
    ]  # [B, C]
    O = torch.masked_select(outputs_mask, one_hot_labels_mask.bool())  # [B, C] -> [B, ]

    avg_drop = torch.maximum(Y - O, torch.zeros_like(Y)) / Y  # [B, ]
    avg_gain = torch.maximum(O - Y, torch.zeros_like(Y)) / (1 - Y)  # [B, ]
    avg_inc = torch.gt(O, Y)  # [B, ]
    # avg_drop = np.sum(np.max((Y-O,np.zeros(Y.shape)),axis=0)/Y)/O.shape[0]
    # avg_inc = np.sum(np.greater(O,Y))/O.shape[0]

    return avg_drop, avg_inc, avg_gain


# def prepare_adi_input(
#     model: CBM_Net,
#     ind_X: torch.Tensor,
#     explain_algorithm_forward: Callable,
#     ind_attr_labels: torch.Tensor,
# ):
#     """
#         model: CBM_Net
#         ind_X: [1, C, H, W]
#         ind_attr_labels: [D]
#         explain_algorithm_forward: Callable
#     """
#     k_val = int(torch.sum(ind_attr_labels).item())
#     torch.nonzero(ind_attr_labels)

#     model.zero_grad()
#     concepts_attribution: torch.Tensor = explain_algorithm_forward(
#         batch_X=ind_X, target=torch.nonzero(ind_attr_labels)
#     )  # [K, 1, H, W]
#     concepts_attribution.sigmoid_()


def concepts_adi(
    model: Callable,
    images: torch.Tensor,
    masked_images: torch.Tensor,
    active_labels: torch.Tensor,
):
    """
    Args:
        model: nn.Module
        images: [1, 3, H, W]
        masked_images: [K, 3, H, W]
        active_labels: [1, D] where sum(active_labels) == K

        where K -> active concept label, D -> concept label

    """
    images = images
    masked_images = masked_images

    concepts_logits: torch.Tensor = model(images)  # [1, D]
    masked_concepts_logits: torch.Tensor = model(masked_images)  # [B, D]

    Y = concepts_logits[:, active_labels[0].bool()]  # [1, D] -> [K, ]
    O = masked_concepts_logits[:, active_labels[0].bool()]  # [K, D] -> [K, ]

    avg_drop = torch.maximum(Y - O, torch.zeros_like(Y)) / Y  # [K, ]
    avg_gain = torch.maximum(O - Y, torch.zeros_like(Y)) / (1 - Y)  # [K, ]
    avg_inc = torch.gt(O, Y)  # [K, ]

    return avg_drop, avg_inc, avg_gain


def energy_point_game(bbox, saliency_map):
    w, h = saliency_map.shape

    empty = np.zeros((w, h))
    for box in bbox:
        empty[box.xslice, box.yslice] = 1
    mask_bbox = saliency_map * empty

    energy_bbox = mask_bbox.sum()
    energy_whole = saliency_map.sum()

    proportion = energy_bbox / energy_whole

    return proportion


### Insetion/Deletion
n_classes = 1000
HW = 224 * 224
transf = transforms.ToPILImage()


def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.data.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = (inp - mean) / std
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)
    # cv2.imwrite(title,np.uint8(inp))
    return inp


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt("../evaluation/synset_words.txt", str, delimiter="\t")
    return " ".join(labels[c].split(",")[0].split()[1:])


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype("float32"))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric:
    def __init__(self, model, mode, step, substrate_fn):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ["del", "ins"]
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        """Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        # HW = img_tensor.shape[2] * img_tensor.shape[2] # image area
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == "del":
            title = "Deletion game"
            ylabel = "Pixels deleted"
            type_name = "del"
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == "ins":
            title = "Insertion game"
            ylabel = "Pixels inserted"
            type_name = "ins"
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(
            np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1
        )
        for i in range(n_steps + 1):
            pred = self.model(start)
            if (i % 5) == 0 and i < 100:
                inp = np.array(transf(start.squeeze()))
                cv2.imwrite(
                    "check_hihd/scorecam_hi_997_" + str(i) + ".png", np.uint8(inp)
                )
            # tensor_imshow(start, title='check_hihd/')
            prob = nn.functional.softmax(pred, dim=1)
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print("{}: {:.3f}".format(get_class_name(cl[0][0]), float(pr[0][0])))
                print("{}: {:.3f}".format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = prob[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            # if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
            if verbose == 1 and save_to and i > (n_steps - 1):
                plt.figure()
                # plt.figure(figsize=(10, 5))
                # plt.subplot(121)
                # plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                # plt.axis('off')
                # tensor_imshow(start)

                # plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[: i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(
                    np.arange(i + 1) / n_steps, 0, scores[: i + 1], alpha=0.4
                )
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                plt.savefig(save_to + type_name + "{:03d}.png".format(i))
                plt.close()
            if i < n_steps:
                coords = salient_order[:, self.step * i : self.step * (i + 1)]
                start.view(1, 3, HW)[0, :, coords.copy()] = finish.view(1, 3, HW)[
                    0, :, coords.copy()
                ]
                # start.detach().cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.detach().cpu().numpy().reshape(1, 3, HW)[0, :, coords]
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        """Efficiently evaluate big batch of images.
        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.
        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc="Predicting labels"):
            preds = self.model(img_batch[i * batch_size : (i + 1) * batch_size]).cpu()
            predictions[i * batch_size : (i + 1) * batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc="Substrate"):
            substrate[j * batch_size : (j + 1) * batch_size] = self.substrate_fn(
                img_batch[j * batch_size : (j + 1) * batch_size]
            )

        if self.mode == "del":
            caption = "Deleting  "
            start = img_batch.clone()
            finish = substrate
        elif self.mode == "ins":
            caption = "Inserting "
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps + 1), desc=caption + "pixels"):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j * batch_size : (j + 1) * batch_size])
                preds = (
                    preds.detach()
                    .cpu()
                    .numpy()[
                        range(batch_size), top[j * batch_size : (j + 1) * batch_size]
                    ]
                )
                scores[i, j * batch_size : (j + 1) * batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i : self.step * (i + 1)]
            start.detach().cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = (
                finish.detach().cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
            )
        print("AUC: {}".format(auc(scores.mean(1))))
        return scores
