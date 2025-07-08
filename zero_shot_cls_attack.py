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
from torchvision import datasets
import torchvision.transforms as transforms

from autoattack import AutoAttack

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution
from utils import *

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    parser.add_argument("--concept-bank", default="concept_banks/multimodal_concept_clip_RN50_cifar10_recurse_1.pkl", type=str, help="Path to the concept bank")
    
    parser.add_argument(
        "--backbone-ckpt", default="model_zoo/clip", type=str, help="Path to the backbone ckpt"
    )
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)

    parser.add_argument("--pcbm-arch", default="pcbm", type=str, help="Bottleneck model architecture")
    parser.add_argument("--pcbm-ckpt", default="data/ckpt/RIVAL_10/CIFAR_10/pcbm_cifar10__clip_RN50__multimodal_concept_clip_RN50_cifar10_recurse_1__lam_0.0002__alpha_0.99__seed_42.ckpt", type=str, help="Path to the PCBM checkpoint")

    
    parser.add_argument("--dataset", default="rival10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument("--eps", default=8/255, type=float)
    
    parser.add_argument('--save-100-local', action='store_true')


    return parser.parse_args()
    
def main(args):
    set_random_seed(args.universal_seed)
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    model.eval()
    
    adversary = AutoAttack(lambda x: model(x), 
                           norm='Linf', 
                           eps=args.eps, 
                           version='standard',
                           verbose=False)

    totall_accuracy_adv = []
    totall_accuracy = []
    for idx, data in tqdm(enumerate(dataset.test_loader), 
                          total=dataset.test_loader.__len__()):
        batch_X, batch_Y = data
        batch_X:torch.Tensor = batch_X.to(args.device)
        batch_Y:torch.Tensor = batch_Y.to(args.device)
        
        # x_adv = adversary.run_standard_evaluation(batch_X, batch_Y, bs=args.batch_size)
        # dict_adv = adversary.run_standard_evaluation_individual(batch_X, batch_Y, bs=args.batch_size)
        totall_accuracy.append((model(batch_X)[0].argmax(-1) == batch_Y).float().mean().item())
        # totall_accuracy_adv.append((model(x_adv, True) == batch_Y).float().mean().item())
    totall_accuracy = np.array(totall_accuracy).mean()
    # totall_accuracy_adv = np.array(totall_accuracy_adv).mean()
    
    print(f"accuracy: {totall_accuracy}")
    # print(f"accuracy: {totall_accuracy}")
    
if __name__ == "__main__":
    args = config()
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    