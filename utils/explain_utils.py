import torch
import torch.nn as nn
import numpy as np

import clip
import open_clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer
from captum.attr import *
from .model_utils import *
from typing import Callable, Union
from scipy.ndimage import gaussian_filter

class layer_grad_cam_vit:
    def __init__(self, forward_func:Callable, 
                    target_layer:nn.Module,
                    batch_first:bool=False,
                    contain_cls_token:bool=True) -> None:
        # forward_func is a packed function from model
        self.forward_func = forward_func
        # # model is used for clearing gradients and adjust behaviors of the model
        # self.model = model
        # target_layer is used for tracking the gradients while forwarding everytime
        self.target_layer = target_layer
        self.batch_first = batch_first
        self.contain_cls_token = contain_cls_token
        
        self.gradients:torch.Tensor = None
        self.activations:torch.Tensor = None
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation))
        self.handles.append(
            target_layer.register_forward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        activation:torch.Tensor = output
        if self.batch_first:
            self.activations = activation.detach()
        else:
            self.activations = activation.detach().permute((1, 0, 2))
        
        if self.contain_cls_token:
            self.activations = self.activations[:, 1:, :]

    def save_gradient(self, module, input, output:torch.Tensor):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        
        def _store_grad(grad:torch.Tensor):
            if self.batch_first:
                self.gradients = grad.detach()
            else:
                self.gradients = grad.detach().permute((1, 0, 2))

            if self.contain_cls_token:
                self.gradients = self.gradients[:, 1:, :]

        output.register_hook(_store_grad)
    

    def attribute(self, batch_X:torch.Tensor, target:Union[torch.Tensor|int], additional_args:dict={}):
        self.gradients:torch.Tensor = None # [B, grid ** 2, F]
        self.activations:torch.Tensor = None # [B, grid ** 2, F]
        
        output = self.forward_func(batch_X, **additional_args)
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]
        if isinstance(target, torch.Tensor):
            loss = output.gather(1, target.view(-1, 1))
            loss.sum().backward()
        else:
            loss = output[:, target]
            loss.backward()
        
        B, F = self.gradients.size(0), self.gradients.size(2)
        
        importance_weights = self.gradients.mean(dim=(1)) # [B, F]
        
        weighted_activations = importance_weights[:, None, :] * self.activations # [B, grid ** 2, F]
        weighted_activations = weighted_activations.permute(0, 2, 1) # [B, F, grid ** 2]
        _grid = int(np.round(np.sqrt(weighted_activations.size(2))))
        weighted_activations = weighted_activations.reshape(B, F, _grid, _grid) # [B, F, grid, grid]

        return weighted_activations
    
class model_explain_algorithm_factory:
    
    @staticmethod
    def saliency_map(forward_func:Callable,
                     model:CBM_Net):
        saliency = Saliency(forward_func)
        return saliency
    
    @staticmethod
    def integrated_gradient(forward_func:Callable,
                            model:CBM_Net):
        integrated_grad = IntegratedGradients(forward_func)
        return integrated_grad
    
    @staticmethod
    def guided_grad_cam(forward_func:Callable,
                        model:CBM_Net):
        raise NotImplementedError
        guided_gradcam = GuidedGradCam(posthoc_concept_net,
                                        getattr(posthoc_concept_net.get_backbone(),
                                                target_layer))
        return guided_gradcam
    
    @staticmethod
    def layer_grad_cam(forward_func:Callable,
                       model:CBM_Net):
        layer_grad_cam = None
        backbone = model.get_backbone()
        if isinstance(model, CLIPWrapper):
            backbone = model.backbone.visual

        if isinstance(backbone, CLIP):
            if isinstance(backbone.visual, ModifiedResNet):
                layer_grad_cam = LayerGradCam(forward_func,
                                                backbone.visual.get_submodule("layer4.2"))

        elif isinstance(backbone, open_clip.model.CLIP):
            if isinstance(backbone.visual, open_clip.model.ModifiedResNet):
                layer_grad_cam = LayerGradCam(forward_func,
                                                backbone.visual.get_submodule("layer4.2"))
            elif isinstance(backbone.visual, open_clip.model.VisionTransformer):
                image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
                last_blocks = image_attn_blocks[-1].ln_1
                layer_grad_cam = LayerGradCam(forward_func,
                                                last_blocks)
            
        elif isinstance(backbone, ResNetBottom):
            layer_grad_cam = LayerGradCam(forward_func,
                                          backbone.get_submodule("features.stage4"))
            # layer_grad_cam = LayerGradCam(posthoc_concept_net,
            #                               backbone.get_submodule("features.stage4.unit2.body.conv2.conv"))
        else:
            raise NotImplementedError
        
        return layer_grad_cam
    
    @staticmethod
    def layer_grad_cam_vit(forward_func:Callable,
                            model:CBM_Net):
        layer_grad_cam = None
        backbone = model.get_backbone()

        # if isinstance(model, spss_pcbm):
        #     last_blocks = model.get_expalainable_component()
        #     layer_grad_cam = layer_grad_cam_vit(forward_func,
        #                                     last_blocks,
        #                                     batch_first=False)
        if isinstance(backbone, CLIP) and isinstance(backbone.visual, VisionTransformer):
            image_attn_blocks = list(backbone.visual.get_submodule("transformer.resblocks").children())
            last_blocks = image_attn_blocks[-1].get_submodule("ln_1")
            layer_grad_cam = layer_grad_cam_vit(forward_func,
                                            last_blocks,
                                            batch_first=False)
        elif isinstance(backbone, open_clip.model.CLIP) and isinstance(backbone.visual, open_clip.model.VisionTransformer):
            image_attn_blocks = list(backbone.visual.get_submodule("transformer.resblocks").children())
            last_blocks = image_attn_blocks[-1].get_submodule("ln_1")
            layer_grad_cam = layer_grad_cam_vit(forward_func,
                                            last_blocks,
                                            batch_first=backbone.visual.transformer.batch_first)
        else:
            raise NotImplementedError
                    
        return layer_grad_cam

    @staticmethod
    def builtin_explain(forward_func:Callable,
                        model:CBM_Net):
        return model
    
class model_explain_algorithm_forward:

    @staticmethod
    def batch_X_expand(batch_X:torch.Tensor,
                       target:Union[torch.Tensor|int]):
        """
            batch_X: [B, C, W, H] or batch_X: [1, C, W, H]
            target: [C', 1]
        """
        # if target.size().__len__() == 2:
        #     expanded_batch_X = batch_X.unsqueeze(1).expand(-1, target.size(1), -1, -1, -1)
        #     expanded_batch_X = expanded_batch_X.reshape(-1, batch_X.size(-3), batch_X.size(-2), batch_X.size(-1))
        #     target = target.view(-1)

        if isinstance(target, torch.Tensor) and batch_X.size(0) == 1:
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)
        
        return batch_X, target
    
    @staticmethod
    def saliency_map(batch_X:torch.Tensor,
                    explain_algorithm:Saliency,
                    target:Union[torch.Tensor|int]):
        expanded_batch_X, target = __class__.batch_X_expand(batch_X, target)
        attributions:torch.Tensor = explain_algorithm.attribute(expanded_batch_X, target=target, abs=False)
        return attributions
    
    @staticmethod
    def integrated_gradient(batch_X:torch.Tensor,
                            explain_algorithm:IntegratedGradients,
                            target:Union[torch.Tensor|int]):
        expanded_batch_X, target = __class__.batch_X_expand(batch_X, target)
        attributions:torch.Tensor = explain_algorithm.attribute(expanded_batch_X, target=target)
        return attributions
    
    @staticmethod
    def guided_grad_cam(batch_X:torch.Tensor,
                        explain_algorithm:GuidedGradCam,
                        target:Union[torch.Tensor|int]):
        expanded_batch_X, target = __class__.batch_X_expand(batch_X, target)
        attributions:torch.Tensor = explain_algorithm.attribute(expanded_batch_X, target=target)
        return attributions
    
    @staticmethod
    def layer_grad_cam(batch_X:torch.Tensor,
                        explain_algorithm:LayerGradCam,
                        target:Union[torch.Tensor|int]):
        expanded_batch_X, target = __class__.batch_X_expand(batch_X, target)
        attributions:torch.Tensor = explain_algorithm.attribute(expanded_batch_X, target=target)
        upsampled_attr = LayerAttribution.interpolate(attributions, batch_X.size()[-2:], interpolate_mode="bicubic")
        return upsampled_attr
    
    @staticmethod
    def layer_grad_cam_vit(batch_X:torch.Tensor,
                        explain_algorithm:layer_grad_cam_vit,
                        target:Union[torch.Tensor|int]):
        return __class__.layer_grad_cam(batch_X = batch_X,
                              explain_algorithm = explain_algorithm,
                              target = target)
    @staticmethod
    def builtin_explain(batch_X:torch.Tensor,
                        explain_algorithm:CBM_Net,
                        target:Union[torch.Tensor|int]):
        expanded_batch_X, target = __class__.batch_X_expand(batch_X, target)
        attributions:torch.Tensor = explain_algorithm.attribute(expanded_batch_X, target=target)

        if batch_X.size()[-2:] != attributions.size()[-2:]:
            upsampled_attr = LayerAttribution.interpolate(attributions, batch_X.size()[-2:], interpolate_mode="bicubic")
            return upsampled_attr
        
        return attributions

class attribution_pooling_forward:
    
    @staticmethod
    def max_pooling_class_wise(batch_X:torch.Tensor,
                               attributions:torch.Tensor,
                               concept_idx:Union[torch.Tensor, int],
                               pcbm_net:CBM_Net):
        if isinstance(concept_idx, int):
            return attributions.squeeze(0)
        
        with torch.no_grad():
            max_concept_idx = pcbm_net(batch_X)[0, concept_idx].argmax()
        return attributions[max_concept_idx]
    
    @staticmethod
    def max_pooling_channel_wise(batch_X:torch.Tensor,
                               attributions:torch.Tensor,
                               concept_idx:Union[torch.Tensor, int],
                               pcbm_net:CBM_Net):
        return attributions.max(0).values
    
    @staticmethod
    def mean_pooling(batch_X:torch.Tensor,
                               attributions:torch.Tensor,
                               concept_idx:Union[torch.Tensor, int],
                               pcbm_net:CBM_Net):
        return attributions.mean(0)

# Author: Felipe Torres Figueroa - felipe.torres@lis-lab.fr
# Modified from: https://github.com/eclique/RISE/blob/master/evaluation.py
# Functions
def gkern(klen:int, 
          nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

# ========================================================================
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

# ========================================================================
# Causal Metrics
class CausalMetric(nn.Module):
    def __init__(self, model:nn.Module, 
                 mode:str, 
                 step:int,
                 substrate_fn:Callable,
                #  HW:int,
                 classes:int):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): model wrapped in MAE to be explained.
            mode (str): 'del' or 'ins' or 'del-mae'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        super().__init__()
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        # self.HW = HW**2
        self.n_classes = classes

    def forward(self, 
                 img_batch:torch.Tensor,
                 exp_batch:torch.Tensor, 
                 batch_size:int, 
                 top:torch.Tensor):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (torch.Tensor): Array containing scores at every step 
            for every image.
        """
        # Baseline parameters
        B, C, H, W = img_batch.size()
        predictions = torch.FloatTensor(B, self.n_classes)
        
        n_steps = (H * W + self.step - 1) // self.step
        # Flatten -> max to min.
        salient_order = torch.argsort(exp_batch.view(-1, H * W), 
                                           dim=1, 
                                           descending=True)
        r = torch.arange(B).view(B, 1)
        assert B % batch_size == 0
        
        # Forwarding
        #predictions = self.model(img_batch)
        #top = torch.argmax(predictions.detach(), dim=-1).cpu()
        scores = torch.zeros((n_steps + 1, B)).cpu()

        # Generating the starting image-reconstruction to forward
        # substrate = torch.zeros_like(img_batch)
        # for j in range(B // batch_size):
            # Masks every slice of the minibatch using the substrate function
        substrate = self.substrate_fn(img_batch)
            # substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])
            
        if 'del' in self.mode:
            start = img_batch.clone().detach()
            finish = substrate.flatten(start_dim=2)
        elif 'ins' in self.mode:
            start = substrate
            finish = img_batch.clone().detach().flatten(start_dim=2)
        
        with torch.no_grad():
            # While not all pixels are changed
            for i in range(n_steps+1):
                # Iterate over batches
                for j in range(B // batch_size):
                # Iterates over minibatches to retrieve the predicted
                # probabilities for masking step i.
                    # Compute new scores
                    preds = self.model(start[j*batch_size:(j+1)*batch_size])
                    preds = nn.functional.softmax(preds, dim=1)
                    
                    # Gets the predicted probabilities of baseline predictions
                    preds = preds[range(batch_size),
                                top[j*batch_size:(j+1)*batch_size]]
                    
                    # Appends probabilities to scores
                    scores[i, j*batch_size:(j+1)*batch_size] = preds.detach().cpu()
                    
                # Change specified number of most salient px to substrate px
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start = start.flatten(start_dim=2)
                start[r, :, coords] = finish[r, :, coords]
                start = start.view(img_batch.size())
        return scores


def concepts_adi(
    model: Callable,
    images: torch.Tensor,
    masked_images: torch.Tensor,
    active_labels: Union[torch.Tensor, int] = None,
    reduction: str = "mean",
):
    """
    Args:
        model: nn.Module
        images: [1, 3, H, W]
        masked_images: [K, 3, H, W]
        active_labels: [1, D] where sum(active_labels) == K

        where K -> active concept label, D -> concept label

    """
    if not hasattr(concepts_adi, "total"):
        concepts_adi.total = 0
        concepts_adi.nan = 0

    images = images
    masked_images = masked_images

    concepts_logits = None
    masked_concepts_logits = None
    with torch.no_grad():
        concepts_logits: torch.Tensor = model(images)  # [1, D]
        masked_concepts_logits: torch.Tensor = model(masked_images)  # [1, D]

    if isinstance(active_labels, torch.Tensor):
        Y = concepts_logits[:, active_labels[0].bool()].sigmoid()  # [1, D] -> [1, K]
        O = masked_concepts_logits[
            :, active_labels[0].bool()
        ].sigmoid()  # [1, D] -> [1, K]
    elif active_labels is not None:
        Y = concepts_logits[:, active_labels].sigmoid()  # [1, D] -> [1, 1]
        O = masked_concepts_logits[:, active_labels].sigmoid()  # [1, D] -> [1, 1]
    else:
        Y = concepts_logits.sigmoid()  # [1, D] -> [1, 1]
        O = masked_concepts_logits.sigmoid()  # [1, D] -> [1, 1]

    avg_drop = torch.maximum(Y - O, torch.zeros_like(Y)) / Y  # [K, ]
    avg_gain = torch.maximum(O - Y, torch.zeros_like(Y)) / (1 - Y)  # [K, ]
    avg_inc = torch.gt(O, Y)  # [K, ]

    concepts_adi.total += 1
    concepts_adi.nan += avg_gain.isnan().any().float().item()
    
    if reduction == "mean":
        return avg_drop.nanmean(), avg_inc.float().nanmean(), avg_gain.nanmean()
    elif reduction is None:
        return avg_drop, avg_inc.float(), avg_gain