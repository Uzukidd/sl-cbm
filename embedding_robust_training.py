import argparse
import random
import clip.model
import numpy as np
import pickle as pkl
import json
import time
from datetime import datetime
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
from asgt import attack_utils


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
    
    parser.add_argument("--train-method", required=True, type=str)
    parser.add_argument("--train-embedding", action='store_true')
    
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--eps", default=0.025, type=float)
    parser.add_argument("--k", default=1e-1, type=float)
    
    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)
    parser.add_argument('--save-100-local', action='store_true')


    return parser.parse_args()


def training_forward_func(loss:torch.Tensor, 
                          model:nn.Module, 
                          optimizer:optim.Optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
class concept_select_func:
    clip10_main_cocnept_classes = {
        "airplane" : "propellers",
        "automobile" : "windshield",
        "bird" : "beak",
        "cat" : "sharp claws",
        "deer" : "antler",
        "dog" : "paws",
        "frog" : "amphibian",
        "horse" : "horseback",
        "ship" : "porthole",
        "truck" : "an engine"
    }
    
    @staticmethod
    def cifar10(args:argparse.Namespace,
            dataset:dataset_collection,
            model_context:model_pipeline):
        class_to_idx = dataset.class_to_idx
        concept_names = model_context.concept_bank.concept_names
        main_cocnept_classes = {class_to_idx[keys]: concept_names.index(vals)
                                                 for keys, vals in __class__.clip10_main_cocnept_classes.items()}
        
        return main_cocnept_classes

class get_embedding_model:
    
    @staticmethod
    def clip(args:argparse.Namespace,
              model_context:model_pipeline):
        posthoc_concept_net = PCBM_Net(model_context=model_context,
                                output_logit=True)
        
        posthoc_concept_net.posthoc_layer.classifier.weight.requires_grad_(False)
        posthoc_concept_net.posthoc_layer.classifier.bias.requires_grad_(False)

        return posthoc_concept_net
    
    @staticmethod
    def open_clip(args:argparse.Namespace,
                  model_context:model_pipeline):
        return __class__.clip(args, 
                              model_context)
        
    @staticmethod
    def resnet18_cub(args:argparse.Namespace,
                     model_context:model_pipeline):
        return model_context.backbone
    
class save_checkpoint:
    
    @staticmethod
    def clip(args:argparse.Namespace, 
             model:PCBM_Net):
        torch.save({"state_dict": model.backbone.state_dict()}, 
                   os.path.join(args.save_path, f"{args.train_method}-{args.backbone_name.replace("/", "-")}.pth"))
        
    @staticmethod
    def open_clip(args:argparse.Namespace, 
                  model:PCBM_Net):
        return __class__.clip(args, 
                              model)
    
    @staticmethod
    def resnet18_cub(args:argparse.Namespace, 
                     model:nn.Module):
        return torch.save({"state_dict": model.state_dict()}, 
                          os.path.join(args.save_path, f"{args.train_method}-{args.backbone_name.replace("/", "-")}.pth"))

def main(args:argparse.Namespace):
    set_random_seed(args.universal_seed)
    concept_bank = load_concept_bank(args)
    backbone, preprocess = load_backbone(args)

    normalizer = transforms.Compose(preprocess.transforms[-1:])
    preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    posthoc_layer = load_pcbm(args)
    dataset = load_dataset(args, preprocess)
    args.data_size = dataset.trainset[0][0].size()
    
    model_context = model_pipeline(concept_bank = concept_bank, 
                   posthoc_layer = posthoc_layer, 
                   preprocess = preprocess, 
                   normalizer = normalizer, 
                   backbone = backbone)
    
    
    # Get a trainable model
    backbone_arch = args.backbone_name.split(":")[0]
    model:nn.Module = getattr(get_embedding_model, backbone_arch)(args = args, model_context=model_context,)
    
    # Get explain algorithm
    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, args.explain_method)(posthoc_concept_net = model)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    
    # Prepare training module
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    classification_attak_func = attack_utils.FGSM(model, nn.CrossEntropyLoss(), eps = args.eps)
    embedding_attak_func = attack_utils.PGD(model, nn.MSELoss(reduction="sum"), 
                                            alpha= args.eps/10, 
                                            random_start=True,
                                            epoch=10, 
                                            eps = args.eps)
    
    targeted_concept_idx = getattr(concept_select_func, args.dataset)(args = args, 
                                                                      dataset = dataset,
                                                                      model_context = model_context)
    print(targeted_concept_idx)

    print(f"data size: {args.data_size}")
    asgt_module = embedding_robust_training(model = model, 
                                        training_forward_func = partial(training_forward_func,
                                                                        model = model,
                                                                        optimizer = optimizer),
                                        classification_attak_func=classification_attak_func,
                                        embedding_attak_func=embedding_attak_func,
                                        explain_func = partial(explain_algorithm_forward, 
                                                                explain_algorithm=explain_algorithm),
                                        targeted_concept_idx = targeted_concept_idx,
                                        k = int(args.data_size[-1] * args.data_size[-2] * args.k),
                                        lam = 1.0,
                                        feature_range= (0.0, 1.0),
                                        device=torch.device(args.device))
    
    # asgt_module = ASGT_Legacy(model = model, 
    #                    training_forward_func = partial(training_forward_func,
    #                                                    model = model,
    #                                                    optimizer = optimizer),
    #                    loss_func = nn.CrossEntropyLoss(),
    #                    attak_func="FGSM",
    #                    explain_func = partial(explain_algorithm_forward, 
    #                                           explain_algorithm=explain_algorithm),
    #                    eps = args.eps,
    #                    k = int(args.data_size[-1] * args.data_size[-2] * args.k),
    #                    lam = 1.0,
    #                    feature_range= (0.0, 1.0),
    #                    device=torch.device(args.device))
    
    # asgt_module.evaluate_model(dataset.train_loader)
    # asgt_module.evaluate_model(dataset.test_loader)
    # asgt_module.evaluate_model_robustness(dataset.test_loader)
    # asgt_module.evaluate_embedding_robustness(dataset.test_loader)

    
    num_epoches = 5
    for epoch in range(num_epoches):
        running_loss = asgt_module.train_one_epoch(dataset.train_loader)
        print(f"Epoch [{epoch + 1}/{num_epoches}], Loss: {running_loss / len(dataset.train_loader):.4f}")
        getattr(save_checkpoint, backbone_arch)(args=args, 
                                                model = model)
        asgt_module.evaluate_model(dataset.train_loader)
        asgt_module.evaluate_model(dataset.test_loader)
        asgt_module.evaluate_model_robustness(dataset.test_loader)
        asgt_module.evaluate_embedding_robustness(dataset.test_loader)
    
    
if __name__ == "__main__":
    args = config()
    args.save_path = os.path.join("./outputs", args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    