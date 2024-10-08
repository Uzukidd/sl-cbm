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
    parser.add_argument("--dataset-path", required=True, type=str, help="Path to the dataset")
    
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
    return parser.parse_args()


class concept_selecting_funcs:
    
    @staticmethod
    def weights_concept_selecting(input_context:model_result, 
                                  model_context:model_pipeline, 
                                  K:int = 5):
        B, F = input_context.concept_projs.size(0), input_context.concept_projs.size(1) 
        
        weights = model_context.posthoc_layer.get_linear_weights() # [C, F]
        topK_values, topk_indices = torch.topk(weights, K, dim=1)  # [C, K]
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
                            eps:float = 0.01):
    batch_X = batch_X.clone().detach().requires_grad_(True)
    batch_X_normalized = model_context.normalization(batch_X)
    embeddings = model_context.backbone.encode_image(batch_X_normalized)
    concept_projs = model_context.posthoc_layer.compute_dist(embeddings)
        
    concept_selecting_func(input_context = model_result(batch_X = batch_X,
                                                        batch_Y = batch_Y,
                                                        embeddings = embeddings,
                                                        concept_projs = concept_projs),
                           model_context = model_context).mean().backward()
    batch_X = (batch_X - eps * batch_X.grad.sign()).clamp_(0.0, 1.0)
    
    return batch_X.detach()


def main(args):
    set_random_seed(args.universal_seed)
    concept_bank = load_concept_bank(args)
    backbone, preprocess = load_backbone(args)
    normalization = transforms.Compose(preprocess.transforms[-1:])
    preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    posthoc_layer = load_pcbm(args)
    trainset, testset, class_to_idx, idx_to_class, train_loader, test_loader = load_dataset(args, preprocess)
    
    model_context = model_pipeline(concept_bank = concept_bank, 
                   posthoc_layer = posthoc_layer, 
                   preprocess = preprocess, 
                   normalization = normalization, 
                   backbone = backbone)
    
    def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
        import torch
        import torchvision
        import matplotlib.pyplot as plt
        
        if comparison_images is not None:
            images = torch.cat((images, comparison_images), dim=3)

        # 使用 torchvision.utils.make_grid 将 64 张图片排列成 8x8 的网格
        grid_img = torchvision.utils.make_grid(images, nrow=8, normalize=True)

        # 转换为 NumPy 格式以便用 matplotlib 显示
        plt.imshow(grid_img.permute(1, 2, 0))  # 转换为 [H, W, C]
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        
    accuracy_ori = []
    accuracy_adv = []

    for idx, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        # print(data.__len__())
        # print(f"x: {data[0].size()}")
        # print(f"y: {data[1].size()}")
        batch_X, batch_Y = data
        batch_X = batch_X.to(args.device)
        batch_Y = batch_Y.to(args.device)
        
        # batch_X_normalized = normalization(batch_X)
        # embeddings = backbone.encode_image(batch_X_normalized)
        # projs = posthoc_layer.compute_dist(embeddings)
        # predicted_Y = posthoc_layer.forward_projs(projs)
        # accuracy = (predicted_Y.argmax(1) == batch_Y).float().mean().item()
        
        # _, topk_indices = torch.topk(projs, 5, dim=1)
        # topk_concept = [[posthoc_layer.names[idx] for idx in row] for row in topk_indices]
        
        # get_topK_concept_logit(args, batch_X,
        #                         batch_Y, 
        #                         normalization, 
        #                         backbone, 
        #                         posthoc_layer)

        batch_X_disturb = cocnept_disturbate(args, batch_X, 
                                batch_Y,
                                model_context,
                                concept_selecting_func=getattr(concept_selecting_funcs, args.cocnept_selecting_func))
        
        accuracy_ori.append(evaluzate_accuracy(args, batch_X,
                                batch_Y, 
                                model_context))
        accuracy_adv.append(evaluzate_accuracy(args, batch_X_disturb,
                                batch_Y, 
                                model_context))
        
        # get_topK_concept_logit(args, batch_X_disturb,
        #                         batch_Y, 
        #                         normalization, 
        #                         backbone, 
        #                         posthoc_layer)
        
        # print(f"embeddings: {embeddings.size()}")
        # print(f"projections: {projs.size()}")
        # print(f"predicted_Y: {predicted_Y.size()}")
        # print(f"accuracy: {accuracy}")
        # show_image(batch_X.detach().cpu(), batch_X_disturb.detach().cpu())
        # import pdb; pdb.set_trace()
        
    print(f"accuracy_ori: {np.array(accuracy_ori).mean()}")
    print(f"accuracy_adv: {np.array(accuracy_adv).mean()}")
    
    
if __name__ == "__main__":
    args = config()
    main(args)
    