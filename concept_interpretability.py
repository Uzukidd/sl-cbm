import argparse
import random
import clip.model
import numpy as np
import pickle as pkl
import json
from tqdm import tqdm
from typing import Tuple, Callable, Union, Dict

import clip
from clip.model import CLIP

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution

from common_utils import *
from attack_utils import *
from model_utils import *
from visual_utils import *


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=24, type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--concept-target", required=True, type=str)
    parser.add_argument("--class-target", default="", type=str)
    
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    return parser.parse_args()

class model_explain_algorithm_factory:
    
    @staticmethod
    def integrated_gradient(args, 
                            posthoc_concept_net:PCBM_Net,):
        from captum.attr import IntegratedGradients
        integrated_grad = IntegratedGradients(posthoc_concept_net)
        return integrated_grad
    
    @staticmethod
    def guided_grad_cam(args, 
                        posthoc_concept_net:PCBM_Net,
                        target_layer:str="layer4"):
        from captum.attr import GuidedGradCam
        guided_gradcam = GuidedGradCam(posthoc_concept_net,
                                        getattr(posthoc_concept_net.get_backbone(),
                                                target_layer))
        return guided_gradcam
    
    @staticmethod
    def layer_grad_cam(args, 
                posthoc_concept_net:PCBM_Net,
                target_layer:str="layer4"):
        from captum.attr import LayerGradCam
        layer_grad_cam = None
        backbone = posthoc_concept_net.backbone
        if isinstance(backbone, CLIP):
            layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                            getattr(backbone.visual,
                                                    target_layer))
        elif isinstance(backbone, ResNetBottom):
            layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                          backbone.get_submodule("features").get_submodule("0").get_submodule("stage4"),
                                          target_layer)
            
        return layer_grad_cam
    
class model_explain_algorithm_forward:
    
    @staticmethod
    def integrated_gradient(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def guided_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def layer_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        upsampled_attr = LayerAttribution.interpolate(attributions, batch_X.size()[-2:], interpolate_mode="bicubic")
        return upsampled_attr

def main(args):
    set_random_seed(args.universal_seed)
    concept_bank = load_concept_bank(args)
    backbone, preprocess = load_backbone(args)
    normalizer = transforms.Compose(preprocess.transforms[-1:])
    preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    posthoc_layer = load_pcbm(args)
    trainset, testset, class_to_idx, idx_to_class, train_loader, test_loader = load_dataset(args, preprocess)
    
    model_context = model_pipeline(concept_bank = concept_bank, 
                   posthoc_layer = posthoc_layer, 
                   preprocess = preprocess, 
                   normalizer = normalizer, 
                   backbone = backbone)
    posthoc_concept_net = PCBM_Net(model_context=model_context)
    
    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, args.explain_method)(args = args, 
                                                                  posthoc_concept_net = posthoc_concept_net)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    
    try:
        targeted_concept_idx = concept_bank.concept_names.index(args.concept_target)
    except:
        targeted_concept_idx = concept_bank.concept_names.index(int(args.concept_target))
        
    print(targeted_concept_idx)
    
    
    for idx, data in tqdm(enumerate(train_loader), 
                          total=train_loader.__len__()):
        batch_X, batch_Y = data
        batch_X:torch.Tensor = batch_X.to(args.device)
        batch_Y:torch.Tensor = batch_Y.to(args.device)
        
        if args.class_target != "" and idx_to_class[batch_Y.item()] != args.class_target:
            continue
        
        if posthoc_concept_net(batch_X, True).item() != batch_Y.item():
            continue
        
        batch_X.requires_grad_(True)
        
        # show_image(batch_X.detach().cpu())
        
        # attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=targeted_concept_idx)
        attributions:torch.Tensor = explain_algorithm_forward(batch_X=batch_X, 
                                                              explain_algorithm=explain_algorithm,
                                                              target=targeted_concept_idx)
        
        
        for i in range(batch_Y.size(0)):
            print(f"ground truth: {idx_to_class[batch_Y[i].item()]}")
        topK_concept_to_name(args, posthoc_concept_net, batch_X)
        viz_attn(batch_X.squeeze().permute((1, 2, 0)).detach().cpu().numpy(),
                 attributions.squeeze(0).mean(0).detach().cpu().numpy())
        # _ = visualization.visualize_image_attr_multiple(attributions.squeeze().permute((1, 2, 0)).detach().cpu().numpy(), 
        #                                     batch_X.squeeze().permute((1, 2, 0)).detach().cpu().numpy(), 
        #                                     signs=["all", 
        #                                            "positive",
        #                                            "positive",
        #                                            "positive",
        #                                            "positive"],
        #                                     titles=[None,
        #                                             None,
        #                                             f"{idx_to_class[batch_Y[i].item()]}-attributions: {args.concept_target}",
        #                                             None,
        #                                             None],
        #                                     methods=["original_image", "heat_map", "blended_heat_map", "masked_image", "alpha_scaling"],)
        
    
    # original_Xs = torch.concat(original_Xs, dim = 0)
    # batch_Ys = torch.concat(batch_Ys, dim = 0)
    # adversarial_Xs = torch.concat(adversarial_Xs, dim = 0)
    
    # evaluate_adversarial_sample(ori_adv_pair(
    #     original_X=(adversarial_Xs - original_Xs),
    #     # adversarial_X=adversarial_Xs,
    # ))
    
if __name__ == "__main__":
    args = config()
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    