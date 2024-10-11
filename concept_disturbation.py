import argparse
import random
import numpy as np
import pickle as pkl
import json
from tqdm import tqdm
from typing import Tuple, Callable, Union, Dict

import clip
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from common_utils import *
from attack_utils import *


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=2024, type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    
    # parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--cocnept-selecting-func", required=True, type=str)
    parser.add_argument("--attack-func", required=True, type=str)
    parser.add_argument("--epsilon", required=True, type=float)


    return parser.parse_args()


class concept_selecting_funcs:
    
    TOPK_WEIGHTS:Tuple[torch.Tensor,
                       torch.Tensor]= None
    
    @staticmethod
    def weights_concept_selecting(input_context:model_result, 
                                  model_context:model_pipeline, 
                                  K:int = 5):
        B, F = input_context.concept_projs.size(0), input_context.concept_projs.size(1) 
        
        if __class__.TOPK_WEIGHTS is None:
            weights = model_context.posthoc_layer.get_linear_weights() # [C, F]
            topK_values, topk_indices = torch.topk(weights, K, dim=1)  # [C, K]
            __class__.TOPK_WEIGHTS = topK_values.detach().clone(), topk_indices.detach().clone()
            assert topK_values.greater(0.0).all(), "Negative weights exist."
            
        topK_values, topk_indices = __class__.TOPK_WEIGHTS
        batch_W = topK_values[input_context.batch_Y, :] #[B, F]
        batch_index = topk_indices[input_context.batch_Y, :]
        batch_indices = torch.arange(B)\
            .view(B, 1)\
            .expand(B, K)
        select_concept_projs = input_context.concept_projs[batch_indices, batch_index]
        
        
        return select_concept_projs.mul(batch_W)
    
    @staticmethod
    def randomK_concept_selecting(input_context:model_result, 
                                  model_context:model_pipeline, 
                                  K:int = 5):
        B, F = input_context.concept_projs.size(0), input_context.concept_projs.size(1) 
        random_indices = torch.ones(B, F)\
            .multinomial(K, replacement=False)
            
        batch_indices = torch.arange(B)\
            .view(B, 1)\
            .expand(B, K)
                
        sampled_projs = input_context.concept_projs[batch_indices, random_indices]
        return sampled_projs

    @staticmethod
    def leastK_concept_selecting(input_context:model_result, 
                                 model_context:model_pipeline, 
                                 K:int = 5):
        topK_values, topk_indices = torch.topk(input_context.concept_projs, K, dim=1, largest=False)
        return topK_values

    @staticmethod
    def topK_concept_selecting(input_context:model_result, 
                               model_context:model_pipeline, 
                               K:int = 5):
        topK_values, topk_indices = torch.topk(input_context.concept_projs, K, dim=1)
        return topK_values


def cocnept_disturbate(args, batch_X:torch.Tensor, 
                            batch_Y:torch.Tensor, 
                            model_context:model_pipeline,
                            concept_selecting_func:Callable[..., torch.Tensor],
                            attack_func:Callable[..., torch.Tensor],
                            eps:float = 0.01):
    
    attack_endflag = False
    index = 0
    original_X = batch_X
    while(not attack_endflag):
        batch_X = batch_X.clone().detach().requires_grad_(True)
        batch_X_normalized = model_context.normalizer(batch_X)
        embeddings = model_context.backbone.encode_image(batch_X_normalized)
        concept_projs = model_context.posthoc_layer.compute_dist(embeddings)
        
        input_context = model_result(batch_X = batch_X,
                                     batch_Y = batch_Y,
                                     embeddings = embeddings,
                                     concept_projs = concept_projs)
        
        adversarial_loss_handle = concept_selecting_func(input_context = input_context,
                                    model_context = model_context,
                                    K = 5)
        
        batch_X, attack_endflag = attack_func(attack_context = adversarial_attack_context(
                                            index = index,
                                            original_X = original_X,
                                            adversarial_loss_handle = adversarial_loss_handle,
                                            input_context = input_context,
                                            model_context = model_context),
                                            eps=args.epsilon)
        index += 1
    
    return batch_X.detach()


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
    
    def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
        import torch
        import torchvision
        import matplotlib.pyplot as plt
        
        if comparison_images is not None:
            images = torch.cat((images, comparison_images), dim=3)

        # 使用 torchvision.utils.make_grid 将 64 张图片排列成 8x8 的网格
        grid_img = torchvision.utils.make_grid(images, nrow=2, normalize=True)

        # 转换为 NumPy 格式以便用 matplotlib 显示
        plt.imshow(grid_img.permute(1, 2, 0))  # 转换为 [H, W, C]
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
    
    original_Xs = []
    batch_Ys = []
    adversarial_Xs = []
    accuracy_ori = []
    accuracy_adv = []

    for idx, data in tqdm(enumerate(test_loader), 
                          total=test_loader.__len__()):
        batch_X, batch_Y = data
        # if not (batch_Y == class_to_idx["bird"]).any():
        #     continue
        
        batch_X = batch_X.to(args.device)
        batch_Y = batch_Y.to(args.device)

        batch_X_disturb = cocnept_disturbate(args, batch_X, 
                                batch_Y,
                                model_context,
                                concept_selecting_func=getattr(concept_selecting_funcs, args.cocnept_selecting_func),
                                attack_func=getattr(adversarial_attack_funcs, args.attack_func))
        
        accuracy_ori.append(evaluzate_accuracy(args, batch_X,
                                batch_Y, 
                                model_context))
        accuracy_adv.append(evaluzate_accuracy(args, batch_X_disturb,
                                batch_Y, 
                                model_context))
        
        # original_Xs.append(batch_X.detach().cpu())
        # batch_Ys.append(batch_Y.detach().cpu())
        # adversarial_Xs.append(batch_X_disturb.detach().cpu())
        
        # get_topK_concept_logit(args, batch_X,
        #                 batch_Y, 
        #                 model_context,)
        
        # get_topK_concept_logit(args, batch_X_disturb,
        #                         batch_Y, 
        #                         model_context,)

        # evaluate_adversarial_sample(ori_adv_pair(
        #     original_X=batch_X,
        #     adversarial_X=batch_X_disturb,
        # ))
        # print(model_context.posthoc_layer.analyze_classifier(k=5))
        # print(((batch_X - batch_X_disturb).abs().detach().cpu() * 5).max())
        # show_image((batch_X - batch_X_disturb).abs().detach().cpu() * 5 , batch_X.detach().cpu())
        # show_image(batch_X.detach().cpu())
        # show_image(batch_X.detach().cpu(), batch_X_disturb.detach().cpu())
        # import pdb; pdb.set_trace()
    
    # original_Xs = torch.concat(original_Xs, dim = 0)
    # batch_Ys = torch.concat(batch_Ys, dim = 0)
    # adversarial_Xs = torch.concat(adversarial_Xs, dim = 0)
    
    # evaluate_adversarial_sample(ori_adv_pair(
    #     original_X=(adversarial_Xs - original_Xs),
    #     # adversarial_X=adversarial_Xs,
    # ))


    print(f"accuracy_ori: {np.array(accuracy_ori).mean()}")
    print(f"accuracy_adv: {np.array(accuracy_adv).mean()}")
    
    
if __name__ == "__main__":
    args = config()
    main(args)
    