import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from clip.model import CLIP
from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, PosthocHybridCBM, get_model
from pcbm.training_tools import load_or_compute_projections

from typing import Tuple, Callable, Union, Optional
from dataclasses import dataclass
from functools import partial

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
    normalizer:transforms.Compose
    backbone:nn.Module
    

class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x

class PCBM_Net(nn.Module):
    def __init__(self, model_context:model_pipeline, 
                 output_class:bool=False, 
                 output_logit:bool=False):
        super().__init__()
        self.normalizer = model_context.normalizer
        self.backbone = model_context.backbone
        self.posthoc_layer = model_context.posthoc_layer
        
        self.output_class = output_class
        self.output_logit = output_logit
        
    def forward(self, input_x:torch.Tensor):
        assert not(self.output_class and self.output_logit)
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        concept_projs = self.posthoc_layer.compute_dist(embeddings)
        
        if self.output_class:
           return self.posthoc_layer.forward_projs(concept_projs).argmax(1)
       
        if self.output_logit:
           return self.posthoc_layer.forward_projs(concept_projs)
        
        return concept_projs
    
    def output_as_class(self, input_x:torch.Tensor):
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        concept_projs = self.posthoc_layer.compute_dist(embeddings)

        return self.posthoc_layer.forward_projs(concept_projs).argmax(1)
       
    
    def get_backbone(self):
        return self.backbone
    
# @dataclass
# class dataset_pipeline:
#     trainset:datasets, testset, class_to_idx, idx_to_class, train_loader, test_loader
