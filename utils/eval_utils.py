import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

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
    if ind_class_name in RIVAL10_features._KEY_FEATURES:
        for attr_name in RIVAL10_features._KEY_FEATURES[ind_class_name]:
            attr_label = RIVAL10_features._ALL_ATTRS.index(attr_name)
            __vis_ind_image(ind_X, attribution, ind_attr_masks, attr_label, f"{ind_class_name}-{attr_name}-{prefix}", save_to)

def save_best_image(ind_class_name:str, 
                   ind_X:torch.Tensor, 
                   attribution:torch.Tensor, 
                   ind_attr_masks:torch.Tensor,
                   best_iou:torch.Tensor,
                   iou:torch.Tensor,
                   save_to:str):
    if ind_class_name in RIVAL10_features._KEY_FEATURES:
        for attr_name in RIVAL10_features._KEY_FEATURES[ind_class_name]:
            attr_label = RIVAL10_features._ALL_ATTRS.index(attr_name)
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