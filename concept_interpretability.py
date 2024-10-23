import argparse
import random
import clip.model
import numpy as np
import pickle as pkl
import json
import time
from tqdm import tqdm
from typing import Tuple, Callable, Union, Dict

import clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution
from explain_utils import layer_grad_cam_vit

from constants import *
from common_utils import *
from attack_utils import *
from model_utils import *
from visual_utils import *


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
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
    
    parser.add_argument('--save-100-local', action='store_true')


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
                        posthoc_concept_net:PCBM_Net):
        from captum.attr import GuidedGradCam
        raise NotImplementedError
        guided_gradcam = GuidedGradCam(posthoc_concept_net,
                                        getattr(posthoc_concept_net.get_backbone(),
                                                target_layer))
        return guided_gradcam
    
    @staticmethod
    def layer_grad_cam(args, 
                posthoc_concept_net:PCBM_Net):
        from captum.attr import LayerGradCam
        layer_grad_cam = None
        backbone = posthoc_concept_net.backbone
        if isinstance(backbone, CLIP):
            if isinstance(backbone.visual, ModifiedResNet):
                layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                                getattr(backbone.visual,
                                                        "layer4")[-1])
            
            # elif isinstance(backbone.visual, VisionTransformer):
            #     image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
            #     # print(image_attn_blocks[0].attn_probs.size())
            #     # image_attn_blocks = backbone.visual.transformer
            #     # image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
            #     # print(image_attn_blocks)
            #     last_blocks = image_attn_blocks[-2]
            #     # print(last_blocks)
            #     layer_grad_cam = LayerGradCam(posthoc_concept_net,
            #                                     last_blocks)
            #     # layer_grad_cam = layer_grad_cam_vit(posthoc_concept_net,
            #     #                                 last_blocks)
            
        elif isinstance(backbone, ResNetBottom):
            layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                          backbone.get_submodule("features").get_submodule("0").get_submodule("stage4") )
            
        return layer_grad_cam
    
    @staticmethod
    def layer_grad_cam_vit(args, 
                posthoc_concept_net:PCBM_Net):
        from captum.attr import LayerGradCam
        layer_grad_cam = None
        backbone = posthoc_concept_net.backbone
        if isinstance(backbone, CLIP) and isinstance(backbone.visual, VisionTransformer):
            image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
            last_blocks = image_attn_blocks[-1].ln_1
            layer_grad_cam = layer_grad_cam_vit(posthoc_concept_net,
                                            last_blocks)
        else:
            raise NotImplementedError
                    
        return layer_grad_cam
    
class model_explain_algorithm_forward:
    
    @staticmethod
    def integrated_gradient(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def guided_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def layer_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        upsampled_attr = LayerAttribution.interpolate(attributions, batch_X.size()[-2:], interpolate_mode="bicubic")
        return upsampled_attr
    
    @staticmethod
    def layer_grad_cam_vit(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        return __class__.layer_grad_cam(batch_X = batch_X,
                              explain_algorithm = explain_algorithm,
                              target = target)



class concept_select_func:
    @staticmethod
    def cifar10(model_context: model_pipeline,
                concept_target:str):
        targeted_concept_idx = model_context.concept_bank.concept_names.index(concept_target)
        return targeted_concept_idx
    
    @staticmethod
    def cub(model_context: model_pipeline,
                concept_target:str):
        if hasattr(CUB_features, concept_target):
            # trick to get the device of a nn.Module
            return torch.arange(getattr(CUB_features, concept_target)[0], getattr(CUB_features, concept_target)[1] + 1)\
                .to(next(model_context.posthoc_layer.parameters()).device)
        
        return model_context.concept_bank.concept_names.index(int(concept_target))
    
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
    targeted_concept_idx = getattr(concept_select_func, args.dataset)(model_context, args.concept_target)
    print(targeted_concept_idx)
    
    count = 0
    for idx, data in tqdm(enumerate(test_loader), 
                          total=test_loader.__len__()):
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
        
        if args.save_100_local:
            if count == 100:
                break
            viz_attn(batch_X,
                    attributions,
                    blur=True,
                    prefix=f"{idx:03d}",
                    save_to=f"data/{args.explain_method}/{args.concept_target}_images")
            try:
                captum_vis_attn(batch_X, 
                                attributions, 
                                title=f"{idx_to_class[batch_Y.item()]}-attributions: {args.concept_target}",
                                save_to=f'data/{args.explain_method}/{args.concept_target}_images/{idx:03d}-captum-image.jpg')
            except:
                pass
               
            count += 1

        else:
            for i in range(batch_Y.size(0)):
                print(f"ground truth: {idx_to_class[batch_Y[i].item()]}")
            topK_concept_to_name(args, posthoc_concept_net, batch_X)
            viz_attn(batch_X,
                    attributions,
                    blur=True,
                    save_to=None)
            captum_vis_attn(batch_X, 
                        attributions, 
                        title=f"{idx_to_class[batch_Y.item()]}-attributions: {args.concept_target}",
                        save_to=None)



        
    
    # original_Xs = torch.concat(original_Xs, dim = 0)
    # batch_Ys = torch.concat(batch_Ys, dim = 0)
    # adversarial_Xs = torch.concat(adversarial_Xs, dim = 0)
    
    # evaluate_adversarial_sample(ori_adv_pair(
    #     original_X=(adversarial_Xs - original_Xs),
    #     # adversarial_X=adversarial_Xs,
    # ))
    
if __name__ == "__main__":
    args = config()
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    