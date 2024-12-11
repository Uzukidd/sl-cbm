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

from abc import ABC, abstractmethod
from .constants import *
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
    
class Tranforms_Wrapper(nn.Module):

    def __init__(self, transform:transforms.Compose, 
                 model:nn.Module):
        super().__init__()
        self.transform = transform
        self.model = model

    def forward(self, input_X):
        if self.transform is not None:
            input_X = self.transform(input_X)

        if self.model is not None:
            input_X = self.model(input_X)

        return input_X


class CLIPWrapper(nn.Module):
    def __init__(self, mtype):
        super(CLIPWrapper, self).__init__()
        self.backbone, self.preprocess = clip.load(mtype, device='cuda', download_root=model_zoo.CLIP)
        self.normalizer = transforms.Compose(self.preprocess.transforms[-1:])
        in_ftrs = self.backbone.encode_image(torch.rand(5,3,224,224).cuda()).shape[1]
        # in_ftrs =  512 if 'ViT' in mtype else 1024
        self.classifier = nn.Linear(in_features=in_ftrs, out_features=10, bias=True)

    def encode(self, x):
        return self.backbone.encode_image(x)
    
    def forward(self, x):
        img_ftrs = self.backbone.encode_image(self.normalizer(x)).float()
        logits = self.classifier(img_ftrs)
        return logits
    
class CBM_Net(ABC, nn.Module):
    
    def __init__(self, model_context:model_pipeline):
        super().__init__()
        self.model_contexts = model_context

        self.output_class = False
        self.output_logit = True
        self.output_embedding = False
        self.output_concepts = False
        
    def output_type(self, type:str):
        self.output_class = False
        self.output_logit = False
        self.output_embedding = False
        self.output_concepts = False
        
        setattr(self, 
                f"output_{type}", 
                True)
    
    # @abstractmethod
    # def get_normalizer(self) -> Union[nn.Module, transforms.Compose]:
    #     pass
    
    # @abstractmethod
    # def get_embedding_encoder(self) -> nn.Module:
    #     pass

    # @abstractmethod
    # def get_cocnept_encoder(self) -> nn.Module:
    #     pass

    # @abstractmethod
    # def get_pcbm_pipeline(self) -> nn.Module:
    #     pass

    # intput -> cocnept projections
    def encode_as_concepts(self, 
                           batch_X:torch.Tensor) -> torch.Tensor:
        concept_encoder = self.get_cocnept_encoder()
        return concept_encoder(batch_X)
    
    # intput -> embedding
    def encode_as_embedding(self, 
                            batch_X:torch.Tensor) -> torch.Tensor:
        backbone = self.get_embedding_encoder()
        return backbone(batch_X)
    
    # intput -> batch logit
    def forward(self, 
                batch_X:torch.Tensor) -> torch.Tensor:
        pcbm = self.get_pcbm_pipeline()
        return pcbm(batch_X)
    
    # # img -> embedding
    # @abstractmethod
    # def embed(self, 
    #           batch_X:torch.Tensor) -> torch.Tensor:
    #     pass
    
    # # embedding -> cocnept projections
    # @abstractmethod
    # def comput_dist(self,
    #                 embeddings:torch.Tensor) -> torch.Tensor:
    #     pass
    
    # # cocnept projections -> batch logit
    # @abstractmethod
    # def forward_projs(self,
    #                   concept_projs:torch.Tensor) -> torch.Tensor:
    #     pass


class PCBM_Net(CBM_Net):
    def __init__(self, model_context:model_pipeline):
        super().__init__(model_context = model_context)
        self.normalizer = model_context.normalizer
        self.backbone = model_context.backbone
        self.posthoc_layer = model_context.posthoc_layer
        
    def forward(self, 
                input_x:torch.Tensor):
        assert (int(self.output_class) + int(self.output_logit) + int(self.output_embedding)) <= 1
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        
        if self.output_embedding:
            return embeddings
        
        concept_projs = self.posthoc_layer.compute_dist(embeddings)
        
        if self.output_class:
            return self.posthoc_layer.forward_projs(concept_projs).argmax(1)
       
        if self.output_logit:
            return self.posthoc_layer.forward_projs(concept_projs)
       
        if self.output_concepts:
            return concept_projs
        
    
    def get_normalizer(self) -> Union[nn.Module, transforms.Compose]:
        return self.normalizer
    
    def get_embedding_encoder(self) -> nn.Module:
        return Tranforms_Wrapper(self.normalizer, self.backbone)

    def get_cocnept_encoder(self) -> nn.Module:
        return Tranforms_Wrapper(self.normalizer, nn.Sequential(
            self.backbone,
            self.posthoc_layer.CAV_layer
        ))

    def get_pcbm_pipeline(self) -> nn.Module:
        return self
    
    def embed(self, 
              input_x:torch.Tensor) -> torch.Tensor:
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        
        return embeddings
    
    def comput_dist(self,
                    embeddings:torch.Tensor) -> torch.Tensor:
        concept_projs = self.posthoc_layer.compute_dist(embeddings)
        
        return concept_projs
    
    def forward_projs(self,
                      concept_projs:torch.Tensor) -> torch.Tensor:
        return self.posthoc_layer.forward_projs(concept_projs)
    
    def output_as_logit(self, 
                        input_x:torch.Tensor) -> torch.Tensor:
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        concept_projs = self.posthoc_layer.compute_dist(embeddings)

        return self.posthoc_layer.forward_projs(concept_projs)
    
    def output_as_class(self, 
                        input_x:torch.Tensor) -> torch.Tensor:
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        concept_projs = self.posthoc_layer.compute_dist(embeddings)

        return self.posthoc_layer.forward_projs(concept_projs).argmax(1)
       
    def get_backbone(self):
        return self.backbone

# @dataclass
# class dataset_pipeline:
#     trainset:datasets, testset, class_to_idx, idx_to_class, train_loader, test_loader
