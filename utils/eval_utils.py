import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from rival10.constants import RIVAL10_constants
from .model_utils import *
from .visual_utils import *
from .constants import *
from typing import Callable

eps=1e-10
def binarize(m):
    m = m.clone()
    # m[m < 0] = 0
    m[torch.isnan(m)] = 0

    max_val = torch.amax(m, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    max_val[max_val < eps] = eps
    m = m / max_val
    m[m>0.5] = 1
    m[m<0.5] = 0
    return m

def attribution_iou(batch_attribution:torch.Tensor, batch_attr_mask:torch.Tensor, eps=1e-10, vis:bool=False, ind_X:torch.Tensor=None):
    """
        args:
            batch_attribution: [B, ...] (non-binarized/non-positive)
            batch_attr_mask: [B, ...] (non-binarized/non-positive)
    """

    binarized_batch_attribution = torch.maximum(batch_attribution, 
                                                batch_attribution.new_zeros(batch_attribution.size()))
    binarized_batch_attribution = binarize(binarized_batch_attribution)

    binarized_batch_attr_mask = batch_attr_mask

    intersection = binarized_batch_attribution * binarized_batch_attr_mask
    intersection = torch.sum(intersection, dim=(1, 2, 3))

    union = torch.sum((binarized_batch_attribution + binarized_batch_attr_mask) > 0, dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + intersection + eps)
    iou = (intersection + eps) / (union + eps)
    return iou.detach(), dice.detach()
    
def __vis_ind_image(ind_X:torch.Tensor, 
                batch_attribution:torch.Tensor,
                batch_attr_mask:torch.Tensor,
                concept:int,
                prefix:str,
                save_to:str):

    
    binarized_batch_attribution = torch.maximum(batch_attribution, 
                                                batch_attribution.new_zeros(batch_attribution.size()))
    binarized_batch_attribution = binarize(binarized_batch_attribution)

    binarized_batch_attr_mask = batch_attr_mask

    viz_attn_multiple(ind_X[0],
        [binarized_batch_attr_mask[concept], binarized_batch_attribution[concept]],
        prefix = prefix,
        save_to = save_to)
    
def save_key_image(ind_class_name:str, 
                   ind_X:torch.Tensor, 
                   attribution:torch.Tensor, 
                   ind_attr_masks:torch.Tensor,
                   prefix:str,
                   save_to:str):
    if ind_class_name in RIVAL10_constants._KEY_FEATURES:
        for attr_name in RIVAL10_constants._KEY_FEATURES[ind_class_name]:
            attr_label = RIVAL10_constants._ALL_ATTRS.index(attr_name)
            __vis_ind_image(ind_X, attribution, ind_attr_masks, attr_label, f"{ind_class_name}-{attr_name}-{prefix}", save_to)

def save_best_image(ind_class_name:str, 
                   ind_X:torch.Tensor, 
                   attribution:torch.Tensor, 
                   ind_attr_masks:torch.Tensor,
                   best_iou:torch.Tensor,
                   iou:torch.Tensor,
                   save_to:str):
    if ind_class_name in RIVAL10_constants._KEY_FEATURES:
        for attr_name in RIVAL10_constants._KEY_FEATURES[ind_class_name]:
            attr_label = RIVAL10_constants._ALL_ATTRS.index(attr_name)
            if best_iou[attr_label] <= iou[attr_label]:
                __vis_ind_image(ind_X, attribution, ind_attr_masks, attr_label, f"{ind_class_name}-{attr_name}-best", save_to)


def interpret_all_concept(args,
                        model:nn.Module,
                        data_loader:DataLoader,
                        explain_algorithm_forward:Callable,
                        explain_concept:torch.Tensor):

    attrwise_best_iou = [None for i in range(10)]
    attrwise_iou = [None for i in range(10)]
    attrwise_amount = [None for i in range(10)]
    
    save_to = os.path.join(args.save_path, "images")

    for idx, data in enumerate(tqdm(data_loader)):
        image:torch.Tensor =  data["img"].to(args.device)
        attr_labels:torch.Tensor = data["attr_labels"].to(args.device)
        attr_masks:torch.Tensor = data["attr_masks"][:, :-1, :, :, :].amax(dim=2, keepdim=True).to(args.device)
        class_label:torch.Tensor = data["og_class_label"].to(args.device)
        class_name:torch.Tensor = data["og_class_name"]

        B, C, W, H = image.size()
        _, K = attr_labels.size()

        for batch_mask in range(B):
            ind_X = image[batch_mask:batch_mask+1]
            ind_attr_labels = attr_labels[batch_mask]
            ind_attr_masks = attr_masks[batch_mask]
            ind_class_name = class_name[batch_mask]
            ind_class_label = class_label[batch_mask].item()

            model.zero_grad()
            attribution = explain_algorithm_forward(
                batch_X = ind_X,
                target = explain_concept
            )
            
            iou, dice =  attribution_iou(attribution.sum(dim=1, keepdim=True), ind_attr_masks, ind_X=ind_X)
            
            if idx < 15:
                save_key_image(ind_class_name, 
                            ind_X,
                            attribution.sum(dim=1, keepdim=True),
                            ind_attr_masks,
                            f"{idx}:{batch_mask}",
                            save_to)
          
            if attrwise_iou[ind_class_label] is None:
                attrwise_iou[ind_class_label] = image.new_zeros(K)
                attrwise_amount[ind_class_label] = image.new_zeros(K)
                attrwise_best_iou[ind_class_label] = image.new_zeros(K)
            
            attrwise_amount[ind_class_label] += ind_attr_labels
            attrwise_iou[ind_class_label] += iou * ind_attr_labels
            
            save_best_image(ind_class_name, 
                            ind_X,
                            attribution.sum(dim=1, keepdim=True),
                            ind_attr_masks,
                            attrwise_best_iou[ind_class_label],
                            iou * ind_attr_labels,
                            save_to)
            attrwise_best_iou[ind_class_label] = torch.max(attrwise_best_iou[ind_class_label], 
                                                           iou)
            
            
    attrwise_iou = torch.stack(attrwise_iou) / torch.stack(attrwise_amount)
    return attrwise_iou

def estimate_top_concepts_accuracy(concept_predictions, concept_labels):

    bs = concept_predictions.shape[0]
    mini_batch_correct_concepts = 0
    mini_batch_total_concepts = 0

    for i in range(bs):

        k_val = int(torch.sum(concept_labels[i]).item())  # active labels for that class
        _, top_gt_indices = torch.topk(concept_labels[i], k=k_val, dim=-1)
        _, top_pred_indices = torch.topk(concept_predictions[i], k=k_val, dim=-1)

        for k in top_pred_indices:
            mini_batch_total_concepts+=1
            if k in top_gt_indices: mini_batch_correct_concepts+=1

    return mini_batch_correct_concepts, mini_batch_total_concepts

def val_one_epoch(val_data_loader:DataLoader, model:CBM_Net, device:torch.device):

    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, data in enumerate(val_data_loader):

            if isinstance(data, list):
                images, class_labels, concept_labels = data
            else:
                images, class_labels, concept_labels = data["img"], data["og_class_label"], data["attr_labels"]
            
            #Loading data and labels to device
            images = images.squeeze().to(device)
            class_labels = class_labels.squeeze().to(device)
            concept_labels = concept_labels.squeeze().to(device)

            #Forward
            class_predictions, concept_predictions , _ = model(images)
            
            # calculate acc per minibatch
            sum_correct_pred += (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
            total_samples += len(class_labels)
            mbcc, mbtc = estimate_top_concepts_accuracy(concept_predictions, concept_labels)
            concept_acc += mbcc
            concept_count += mbtc

    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    return acc, total_concept_acc