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
    
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument('--save-100-local', action='store_true')


    return parser.parse_args()
    
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
    
    totall_accuracy = []
    for idx, data in tqdm(enumerate(test_loader), 
                          total=test_loader.__len__()):
        batch_X, batch_Y = data
        batch_X:torch.Tensor = batch_X.to(args.device)
        batch_Y:torch.Tensor = batch_Y.to(args.device)
        
        totall_accuracy.append((posthoc_concept_net(batch_X, True) == batch_Y).float().mean().item())
    totall_accuracy = np.array(totall_accuracy).mean()
    
    print(f"accuracy: {totall_accuracy}")
    
if __name__ == "__main__":
    args = config()
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    