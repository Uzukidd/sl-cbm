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
from open_clip_train.train import train_one_epoch, evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms

from autoattack import AutoAttack

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution
from utils import *
from asgt import ASGT


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    # parser.add_argument("--train-method", required=True, type=str)
    
    parser.add_argument("--eps", default=0.025, type=float)
    parser.add_argument("--k", default=1e-1, type=float)
    
    parser.add_argument('--save-100-local', action='store_true')


    return parser.parse_args()

def training_forward_func(loss:torch.Tensor, model:nn.Module, optimizer:optim.Optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class get_classification_model:
    
    @staticmethod
    def clip(model_context:model_pipeline):
        return PCBM_Net(model_context=model_context,
                                   output_logit=True)
    @staticmethod
    def open_clip(model_context:model_pipeline):
        return __class__.clip(model_context)
        
    @staticmethod
    def resnet18_cub(model_context:model_pipeline):
        return model_context.backbone


def main(args):
    set_random_seed(args.universal_seed)
    concept_bank = load_concept_bank(args)
    backbone, preprocess = load_backbone(args, full_load=True)

    normalizer = transforms.Compose(preprocess.transforms[-1:])
    preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    posthoc_layer = load_pcbm(args)
    trainset, testset, class_to_idx, idx_to_class, train_loader, test_loader = load_dataset(args, preprocess)
    
    model_context = model_pipeline(concept_bank = concept_bank, 
                   posthoc_layer = posthoc_layer, 
                   preprocess = preprocess, 
                   normalizer = normalizer, 
                   backbone = backbone)
    
        
    backbone_arch = args.backbone_name.split(":")[0]
    posthoc_concept_net = getattr(get_classification_model, backbone_arch)(model_context=model_context,)
    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, args.explain_method)(args = args, 
                                                                  posthoc_concept_net = posthoc_concept_net)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    

    
    optimizer = optim.Adam(posthoc_concept_net.parameters(), lr=1e-4)
    asgt_module = ASGT(model = posthoc_concept_net, 
                       training_forward_func = partial(training_forward_func,
                                                       model = posthoc_concept_net,
                                                       optimizer = optimizer),
                       loss_func = nn.CrossEntropyLoss(),
                       attak_func="FGSM",
                       explain_func = partial(explain_algorithm_forward, 
                                              explain_algorithm=explain_algorithm),
                       eps = args.eps,
                       k = int(224 * 224 * args.k),
                       lam = 1.0,
                        feature_range= (0.0, 1.0),
                       device=torch.device(args.device))
    
    asgt_module.evaluate_model(train_loader)
    asgt_module.evaluate_model(test_loader)
    asgt_module.evaluate_model_robustness(test_loader)
    
    if hasattr(posthoc_concept_net, "posthoc_layer"):
        posthoc_concept_net.posthoc_layer.classifier.weight.requires_grad_(False)
        posthoc_concept_net.posthoc_layer.classifier.bias.requires_grad_(False)
        
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = asgt_module.train_one_epoch(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        torch.save({"state_dict": posthoc_concept_net.backbone.state_dict()}, f"./robust_{args.backbone_name.replace("/", "-")}.pth")
        asgt_module.evaluate_model(train_loader)
        asgt_module.evaluate_model(test_loader)
        robustness = asgt_module.evaluate_model_robustness(test_loader)
    
    
if __name__ == "__main__":
    args = config()
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    