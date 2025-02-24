import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from rival10.constants import RIVAL10_constants
from .model_utils import *
from .visual_utils import *
from .constants import *
from .explain_utils import *
from .common_utils import *
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
    attribution_area = torch.sum(binarized_batch_attribution > 0, dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + intersection + eps)
    iou = (intersection + eps) / (union + eps)
    prec_iou = (intersection + eps) / (attribution_area + eps)

    return iou.detach(), dice.detach(), prec_iou.detach()
    
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
    
def collect_class_attribution(batch_attribution:torch.Tensor, 
                              concepts_indices:torch.Tensor,
                              concepts_weights:torch.Tesnor,
                              label:int):
    """
        args:
            batch_attribution: [B, ...] (non-binarized/non-positive)
            batch_label: [B, 1]
    """
    binarized_batch_attribution = torch.maximum(batch_attribution, 
                                                batch_attribution.new_zeros(batch_attribution.size()))
    binarized_batch_attribution = binarize(binarized_batch_attribution)
    
    class_indices = concepts_indices[label] #[K]
    class_weights = concepts_weights[label] #[K]

    selected_attribution = batch_attribution[class_indices]
    weighted_attribution = torch.sum(
        selected_attribution * class_weights,
        dim=0,
        keepdim=True
    )
    
    return weighted_attribution
    
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

__ALL_METRIC__ = ["iou", "dice", "prec_iou"]
def interpret_all_concept(args,
                        model:nn.Module,
                        data_loader:DataLoader,
                        explain_algorithm_forward:Callable,
                        explain_concept:torch.Tensor):
    attrwise_metric = { }
    for metric in __ALL_METRIC__:
        attrwise_metric[metric] = {
            "best": [None for i in range(10)],
            "val": [None for i in range(10)],
            "amount": [None for i in range(10)],
        }
    
    save_to = os.path.join(args.save_path, "images", "concept")
    
    topK_concepts_idx:torch.Tensor = None
    topK_concepts_weights:torch.Tensor = None
    
    if hasattr(model, "get_topK_concepts"):
        topK_concepts_idx, topK_concepts_weights = model.get_topK_concepts()

    for idx, data in enumerate(tqdm(data_loader)):
        image:torch.Tensor =  data["img"].to(args.device)
        attr_labels:torch.Tensor = data["attr_labels"].to(args.device)
        attr_masks:torch.Tensor = data["attr_masks"][:, :-1, :, :, :].amax(dim=2, keepdim=True).to(args.device)
        class_label:torch.Tensor = data["og_class_label"].to(args.device)
        class_name:torch.Tensor = data["og_class_name"]

        B, C, W, H = image.size()
        _, K = attr_labels.size()
        
        with torch.no_grad():
            _, concept_predictions , _ = model(image)

        for batch_mask in range(B):
            ind_X = image[batch_mask:batch_mask+1]
            ind_attr_labels = attr_labels[batch_mask]
            ind_attr_masks = attr_masks[batch_mask]
            ind_class_name = class_name[batch_mask]
            ind_class_label = class_label[batch_mask].item()

            k_val = int(torch.sum(ind_attr_labels).item())  # active labels for that class
            valid_concepts_mask = torch.zeros_like(ind_attr_labels)
            _, top_pred_indices = torch.topk(concept_predictions[batch_mask], k=k_val, dim=-1)
            valid_concepts_mask[top_pred_indices] = 1
            valid_concepts_mask = valid_concepts_mask & ind_attr_labels


            model.zero_grad()
            attribution = explain_algorithm_forward(
                batch_X = ind_X,
                target = explain_concept
            ) #[18, 1, H, W]
            res_dict = {}
            res_dict["iou"], res_dict["dice"], res_dict["prec_iou"] =  attribution_iou(attribution.sum(dim=1, keepdim=True), ind_attr_masks, ind_X=ind_X)
            
            if idx < 15:
                save_key_image(ind_class_name, 
                            ind_X,
                            attribution.sum(dim=1, keepdim=True),
                            ind_attr_masks,
                            f"{idx}:{batch_mask}",
                            save_to)
                            
            for metric in __ALL_METRIC__:
                attrwise_metric[metric]
                if attrwise_metric[metric]["val"][ind_class_label] is None:
                    attrwise_metric[metric]["val"][ind_class_label] = image.new_zeros(K)
                    attrwise_metric[metric]["amount"][ind_class_label] = image.new_zeros(K)
                    attrwise_metric[metric]["best"][ind_class_label] = image.new_zeros(K)
            
                attrwise_metric[metric]["amount"][ind_class_label] += ind_attr_labels
                attrwise_metric[metric]["val"][ind_class_label] += res_dict[metric] * valid_concepts_mask

                attrwise_metric[metric]["best"][ind_class_label] = torch.max(attrwise_metric[metric]["best"][ind_class_label], 
                                                           res_dict[metric] * valid_concepts_mask)
            
            save_best_image(ind_class_name, 
                            ind_X,
                            attribution.sum(dim=1, keepdim=True),
                            ind_attr_masks,
                            attrwise_metric["iou"]["best"][ind_class_label],
                            res_dict["iou"] * ind_attr_labels,
                            save_to)

            
    for metric in __ALL_METRIC__:
        attrwise_metric[metric]["val"] = torch.stack(attrwise_metric[metric]["val"]) / torch.stack(attrwise_metric[metric]["amount"])
    return {
        metric: attrwise_metric[metric]["val"] for metric in __ALL_METRIC__
    }
    
def interpret_class(args,
                    model:nn.Module,
                    data_loader:DataLoader):
    attrwise_metric = { }
    for metric in __ALL_METRIC__:
        attrwise_metric[metric] = {
            "best": [None for i in range(10)],
            "val": [None for i in range(10)],
            "amount": [None for i in range(10)],
        }
    
    save_to = os.path.join(args.save_path, "images", "class")

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
            attribution = model.attribute_class(
                batch_X = ind_X,
            )
            
            # res_dict = {}
            # res_dict["iou"], res_dict["dice"], res_dict["prec_iou"] =  attribution_iou(attribution(dim=1, keepdim=True), ind_attr_masks, ind_X=ind_X)
            
            if idx < 15:
                __vis_ind_image(ind_X, attribution, ind_attr_masks, -1, f"{idx}-{batch_mask}", save_to)
                            
            # for metric in __ALL_METRIC__:
            #     attrwise_metric[metric]
            #     if attrwise_metric[metric]["val"][ind_class_label] is None:
            #         attrwise_metric[metric]["val"][ind_class_label] = 0
            #         attrwise_metric[metric]["amount"][ind_class_label] = 0
            #         attrwise_metric[metric]["best"][ind_class_label] = 0
                
            #     attrwise_metric[metric]["amount"][ind_class_label] += ind_attr_labels
            #     attrwise_metric[metric]["val"][ind_class_label] += res_dict[metric]

            #     attrwise_metric[metric]["best"][ind_class_label] = torch.max(attrwise_metric[metric]["best"][ind_class_label], 
            #                                                res_dict[metric])
            
            # save_best_image(ind_class_name, 
            #                 ind_X,
            #                 attribution.sum(dim=1, keepdim=True),
            #                 ind_attr_masks,
            #                 attrwise_metric["iou"]["best"][ind_class_label],
            #                 res_dict["iou"] * ind_attr_labels,
            #                 save_to)

            
    # for metric in __ALL_METRIC__:
    #     attrwise_metric[metric]["val"] = torch.stack(attrwise_metric[metric]["val"]) / torch.stack(attrwise_metric[metric]["amount"])
    # return {
    #     metric: attrwise_metric[metric]["val"] for metric in __ALL_METRIC__
    # }

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
            images, class_labels, concept_labels, use_concept_labels = None, None, None, None
            if isinstance(data, list):
                if data.__len__() == 3:
                    images, class_labels, concept_labels = data
                else:
                    images, class_labels, concept_labels, use_concept_labels = data
            else:
                images, class_labels, concept_labels = data["img"], data["og_class_label"], data["attr_labels"]
            
            #Loading data and labels to device
            images = images.squeeze().to(device)
            class_labels = class_labels.squeeze().to(device)
            concept_labels = concept_labels.squeeze().to(device)
            if use_concept_labels is not None:
                use_concept_labels = use_concept_labels.squeeze().to(device)

            if class_labels.size().__len__() == 2:
                class_labels = torch.reshape(class_labels,(class_labels.shape[0]*2,1)).squeeze()
                concept_labels = torch.reshape(concept_labels,(concept_labels.shape[0]*2,18))
                if use_concept_labels is not None:
                    use_concept_labels = torch.reshape(use_concept_labels,(use_concept_labels.shape[0]*2,1)).squeeze()

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


def eval_attribution_alignment(args, model:CBM_Net, dataset:dataset_collection, concept_bank:ConceptBank, explain_method:str):

    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, 
                                                    explain_method)(forward_func=model.encode_as_concepts,
                                                                        model = model)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, explain_method)
    # attribution_pooling:Callable[..., torch.Tensor] = getattr(attribution_pooling_forward, args.concept_pooling)
    explain_concept:torch.Tensor = torch.arange(0, concept_bank.concept_info.concept_names.__len__()).to(args.device)
    
    # Start Rival attrbution alignment evaluation
    interpret_class(args,
                    model,
                    dataset.test_loader, )
    res_dict = interpret_all_concept(args, model,
                            dataset.test_loader, 
                            partial(explain_algorithm_forward, explain_algorithm = explain_algorithm),
                            explain_concept)
    
    for metric in ["iou", "dice", "prec_iou"]:
        attrwise_metric = res_dict[metric]
        torch.save(attrwise_metric.detach().cpu(), os.path.join(args.save_path, f"{metric}_info.pt"))
        args.logger.info(f"--------{metric}\n\n")
        for label, class_name in enumerate(RIVAL10_constants._ALL_CLASSNAMES):
            args.logger.info(f"{class_name}:")
            for name, iou in zip(concept_bank.concept_info.concept_names, attrwise_metric[label]):
                args.logger.info(f" - {name}: {iou:.4f}")

        args.logger.info(f"totall ({attrwise_metric.nanmean():.4f}):")
        for ind, concepts_name in enumerate(concept_bank.concept_info.concept_names):
            args.logger.info(f" - {concepts_name}: {attrwise_metric[:, ind].nanmean():.4f}")