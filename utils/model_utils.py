import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

import clip
from clip.model import CLIP as clip_model_CLIP
import open_clip
from open_clip.model import CLIP as open_clip_model_CLIP

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, PosthocHybridCBM, CAV, get_model
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
    
    def __init__(self):
        super().__init__()

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


    @abstractmethod
    def get_backbone(self) -> nn.Module:
        pass

    @abstractmethod
    # intput -> cocnept projections
    def encode_as_concepts(self, 
                           batch_X:torch.Tensor) -> torch.Tensor:
        pass
    
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
    
    # cocnept projections -> batch logit
    @abstractmethod
    def forward_projs(self,
                      concept_projs:torch.Tensor) -> torch.Tensor:
        pass


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

# CLIP VL-CBM
class clip_cbm(CBM_Net):

    TRAINABLE_COMPONENTS = ["classifier"]

    def __init__(self, normalizer, 
                 concept_bank:ConceptBank, 
                 backbone:Union[open_clip_model_CLIP, clip_model_CLIP],
                 num_of_classes:int=10
                 ):
        super().__init__()

        self.concept_bank = concept_bank
        self.normalizer = normalizer
        self.backbone:Union[open_clip_model_CLIP, clip_model_CLIP] = backbone

        self.CAV_layer = CAV(self.concept_bank.vectors, 
                             self.concept_bank.intercepts, 
                             self.concept_bank.norms,
                             self.concept_bank.concept_names.copy())
        self.num_of_concepts = self.concept_bank.concept_names.__len__()
        self.num_of_classes = num_of_classes
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_of_concepts),
            nn.Linear(self.num_of_concepts, self.num_of_classes),
        )

    def train(self, mode=True):
        for p in self.backbone.parameters(): p.requires_grad=not mode
        for p in self.CAV_layer.parameters(): p.requires_grad=not mode
        super().train(mode)
    
    def set_weights(self, weights:torch.Tensor, bias:torch.Tensor):
        self.classifier[1].weight.data = torch.tensor(weights).to(self.classifier[1].weight.device)
        self.classifier[1].bias.data = torch.tensor(bias).to(self.classifier[1].weight.device)
        return True

    def state_dict(self):
        return {k:v for k, v in super().state_dict().items() if k.split(".")[0] in self.TRAINABLE_COMPONENTS}
    
    def get_backbone(self) -> open_clip_model_CLIP:
        return self.backbone
    
    def forward(self, 
                batch_X:torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_X.size()

        images = self.normalizer(batch_X)

        visual_projection = self.backbone.encode_image(images)
        concept_activations = self.CAV_layer(visual_projection)

        return self.classifier(concept_activations)
    
    def encode_as_concepts(self, 
                           batch_X:torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_X.size()

        images = self.normalizer(batch_X)

        visual_projection = self.backbone.encode_image(images)
        concept_activations = self.CAV_layer(visual_projection)

        return concept_activations
    
    def compute_dist(self, 
                    batch_X:torch.Tensor) -> torch.Tensor:
        return self.encode_as_concepts(batch_X)

    def forward_projs(self,
                      concept_projs:torch.Tensor) -> torch.Tensor:
        return self.classifier(concept_projs)

# Contrastive Semi-Supervised (CSS) VL-CBM
class css_cbm(CBM_Net):

    TRAINABLE_COMPONENTS = ["concept_projection", 
                           "classifier"]

    def __init__(self, normalizer, 
                 concept_bank:ConceptBank, 
                 backbone:open_clip_model_CLIP,
                 num_of_classes:int=10
                 ):
        super().__init__()

        self.concept_bank = concept_bank
        self.normalizer = normalizer
        self.backbone:open_clip_model_CLIP = backbone
        

        assert hasattr(self.backbone.visual, "output_tokens")
        self.backbone.visual.output_tokens = True
        

        self.CAV_layer = CAV(self.concept_bank.vectors, 
                             self.concept_bank.intercepts, 
                             self.concept_bank.norms,
                             self.concept_bank.concept_names.copy())
        self.num_of_concepts = self.concept_bank.concept_names.__len__()
        self.num_of_classes = num_of_classes
        
        self.concept_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, self.num_of_concepts)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_of_concepts),
            nn.Linear(self.num_of_concepts, self.num_of_classes),
        )

    def train(self, mode=True):
        for p in self.backbone.parameters(): p.requires_grad=not mode
        for p in self.CAV_layer.parameters(): p.requires_grad=not mode
        super().train(mode)

    def set_weights(self, weights:torch.Tensor, bias:torch.Tensor):
        self.classifier[1].weight.data = torch.tensor(weights).to(self.classifier[1].weight.device)
        self.classifier[1].bias.data = torch.tensor(bias).to(self.classifier[1].weight.device)
        return True

    def state_dict(self):
        return {k:v for k, v in super().state_dict().items() if k.split(".")[0] in self.TRAINABLE_COMPONENTS}
    
    def get_backbone(self) -> open_clip_model_CLIP:
        return self.backbone
    
    def forward(self, image_pairs):
        bs, imgs, channels, h, w = image_pairs.shape
        images = torch.reshape(image_pairs, 
                               (bs*imgs, channels, h, w))
        images = self.normalizer(images)

        visual_projection, visual_patches = self.backbone.encode_image(images)
        concept_projections = self.concept_projection(torch.mean(visual_patches, 
                                                                 dim=1))

        concept_activations = self.CAV_layer(F.normalize(visual_projection, 
                                                         dim=-1))
        concepts = concept_activations + concept_projections
        #     (bs*2,18)         (bs*2,10)
        return F.sigmoid(concepts), self.classifier(concepts)
    
    def direct_encode_as_concepts(self, 
                           batch_X:torch.Tensor) -> torch.Tensor:
        images = self.normalizer(batch_X)

        visual_projection, _ = self.backbone.encode_image(images)

        concept_activations = self.CAV_layer(F.normalize(visual_projection, 
                                                         dim=-1))
        concepts = concept_activations
        #     (bs*2,18)
        return F.sigmoid(concepts)
    
    def encode_as_concepts(self, 
                           batch_X:torch.Tensor) -> torch.Tensor:
        images = self.normalizer(batch_X)

        visual_projection, visual_patches = self.backbone.encode_image(images)
        concept_projections = self.concept_projection(torch.mean(visual_patches, 
                                                                 dim=1))

        concept_activations = self.CAV_layer(F.normalize(visual_projection, 
                                                         dim=-1))
        concepts = concept_activations + concept_projections
        #     (bs*2,18) 
        return F.sigmoid(concepts)

    def forward_projs(self,
                      concept_projs:torch.Tensor) -> torch.Tensor:
        return self.classifier(concept_projs)

# Modified from https://github.com/billpsomas/simpool/blob/master/sp.py
# Original author: Bill Psomas
class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        
        if gamma is not None:
            self.gamma = torch.tensor([gamma], device='cuda')
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0], device='cuda'))
        self.eps = torch.tensor([1e-6], device='cuda')

        self.gamma = gamma
        self.use_beta = use_beta

    def prepare_input(self, x):
        if len(x.shape) == 3: # Transformer
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2) # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1) # (B, d) -> (B, 1, d)
            return gap_cls, x
        if len(x.shape) == 4: # CNN
            # Input tensor dimensions:
            # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
            B, d, H, W = x.shape
            gap_cls = x.mean([-2, -1]) # (B, d, H, W) -> (B, d)
            x = x.reshape(B, d, H*W).permute(0, 2, 1) # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1) # (B, d) -> (B, 1, d)
            return gap_cls, x
        else:
            raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")

    def forward(self, x):
        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape

        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1, 3) # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1, 3) # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)
        
        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1, 3) # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)

        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)

        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma), 1/self.gamma) # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation 
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)        
        
        return x.squeeze()