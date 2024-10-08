import argparse
import random
import numpy as np
import pickle as pkl
import json
from tqdm import tqdm
from typing import Tuple, Callable, Union, Optional
from dataclasses import dataclass


import clip
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model
from pcbm.training_tools import load_or_compute_projections

@dataclass
class model_result:
    batch_X:Optional[torch.Tensor] = None
    batch_Y:Optional[torch.Tensor] = None
    embeddings:Optional[torch.Tensor] = None
    concept_projs:Optional[torch.Tensor] = None
    batch_Y_predicted:Optional[torch.Tensor] = None


@dataclass
class model_pipeline:
    concept_bank:ConceptBank
    posthoc_layer:PosthocLinearCBM
    preprocess:transforms.Compose
    normalization:transforms.Compose
    backbone:nn.Module


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(args, preprocess:transforms.Compose):
    trainset = datasets.CIFAR10(root=args.dataset_path, train=True,
                                download=False, transform=preprocess)
    testset = datasets.CIFAR10(root=args.dataset_path, train=False,
                                download=False, transform=preprocess)
    classes = trainset.classes
    class_to_idx = {c: i for (i,c) in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers)
    
    return trainset, testset, class_to_idx, idx_to_class, train_loader, test_loader

def load_concept_bank(args) -> ConceptBank:
    all_concepts = pkl.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)
    
    return concept_bank

def load_backbone(args) -> Tuple[nn.Module, transforms.Compose]:

    clip_backbone_name = args.backbone_name.split(":")[1]
    backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.backbone_ckpt)
    backbone = backbone.eval()\
                .float()\
                .to(args.device)\
                .eval()
    
    return backbone, preprocess

def load_pcbm(args) -> PosthocLinearCBM:
    posthoc_layer:PosthocLinearCBM = torch.load(args.pcbm_ckpt, map_location=args.device)
    # print(posthoc_layer.analyze_classifier(k=5))
    # print(posthoc_layer.names)
    # print(posthoc_layer.names.__len__())
    return posthoc_layer

def get_topK_concept_logit(args, batch_X:torch.Tensor, 
                            batch_Y:torch.Tensor,
                            model_context:model_pipeline,
                            K:int = 5,):
    
    batch_X_normalized = model_context.normalization(batch_X)
    embeddings = model_context.backbone.encode_image(batch_X_normalized)
    projs = model_context.posthoc_layer.compute_dist(embeddings)
    predicted_Y = model_context.posthoc_layer.forward_projs(projs)
    accuracy = (predicted_Y.argmax(1) == batch_Y).float().mean().item()
    
    topk_values, topk_indices = torch.topk(projs, 5, dim=1)
    topk_concept = [{model_context.posthoc_layer.names[idx]:round(float(val), 2) for idx, val in zip(irow, vrow)} for irow, vrow in zip(topk_indices, topk_values)]
    classification_res = [f"{model_context.posthoc_layer.idx_to_class[Y.item()]} -> {model_context.posthoc_layer.idx_to_class[Y_hat.item()]}" for Y, Y_hat in zip(batch_Y, predicted_Y.argmax(1))]
    print(f"top (K = {K}) concepts: {json.dumps(topk_concept, indent=4)}")
    print(f"classification result: {json.dumps(classification_res, indent=4)}")
    print(f"accuracy: {accuracy}")

def evaluzate_accuracy(args, batch_X:torch.Tensor, 
                            batch_Y:torch.Tensor,
                            model_context:model_pipeline,):
    batch_X_normalized = model_context.normalization(batch_X)
    embeddings = model_context.backbone.encode_image(batch_X_normalized)
    projs = model_context.posthoc_layer.compute_dist(embeddings)
    predicted_Y = model_context.posthoc_layer.forward_projs(projs)
    accuracy = (predicted_Y.argmax(1) == batch_Y).float().mean().item()
    
    return accuracy