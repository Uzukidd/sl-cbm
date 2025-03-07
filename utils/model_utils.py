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
from collections import OrderedDict


@dataclass
class model_result:
    batch_X: Optional[torch.Tensor] = None
    batch_Y: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    concept_projs: Optional[torch.Tensor] = None
    batch_Y_predicted: Optional[torch.Tensor] = None


@dataclass
class model_pipeline:
    concept_bank: ConceptBank
    preprocess: transforms.Compose
    normalizer: transforms.Compose
    backbone: nn.Module


# class ResNetBottom(nn.Module):
#     def __init__(self, original_model):
#         super(ResNetBottom, self).__init__()
#         self.features = nn.Sequential(*list(original_model.children())[:-1])

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return x


class ResNetBottom(nn.Module):
    def __init__(self, original_model: nn.Module):
        super(ResNetBottom, self).__init__()
        self.features = list(original_model.children())[0]
        self.classifier: nn.Linear = list(original_model.children())[1]

        self.final_pool = list(self.features.children())[-1]
        self.features = nn.Sequential(
            OrderedDict(list(self.features.named_children())[:-1])
        )
        self.output_visual_patches = False

    def forward(self, x):
        visual_patches = self.features(x)  # [B, D, H, W]
        embedding = self.final_pool(visual_patches)  # [B, D, 1, 1]
        embedding = torch.flatten(embedding, 1)  # [B, D]

        if self.output_visual_patches:
            return embedding, visual_patches.permute((0, 2, 3, 1))
        else:
            return embedding


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


class Tranforms_Wrapper(nn.Module):
    def __init__(self, transform: transforms.Compose, model: nn.Module):
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
        self.backbone, self.preprocess = clip.load(
            mtype, device="cuda", download_root=model_zoo.CLIP
        )
        self.normalizer = transforms.Compose(self.preprocess.transforms[-1:])
        in_ftrs = self.backbone.encode_image(torch.rand(5, 3, 224, 224).cuda()).shape[1]
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

    def output_type(self, type: str):
        self.output_class = False
        self.output_logit = False
        self.output_embedding = False
        self.output_concepts = False

        setattr(self, f"output_{type}", True)

    def attribute(
        self,
        batch_X: torch.Tensor,
        target: Union[torch.Tensor | int],
        additional_args: dict = {},
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        pass

    @abstractmethod
    # intput -> cocnept projections
    def encode_as_concepts(self, batch_X: torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    # intput -> embedding
    def encode_as_embedding(self, batch_X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # input -> concept logit, class logit, patch concepts (Optional)
    def forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        pcbm = self.get_pcbm_pipeline()
        return pcbm(batch_X)

    # cocnept projections -> batch logit
    @abstractmethod
    def forward_projs(self, concept_projs: torch.Tensor) -> torch.Tensor:
        pass

    # ------------
    # Getter
    # ------------
    def get_num_classes(self):
        raise NotImplementedError

    def get_num_concepts(self):
        raise NotImplementedError

    # ------------
    # Attribution
    # ------------

    # Get concepts contribution
    def get_topK_concepts(self, K: int = 5):
        """
        Args:
            K:int

        Returns:
            top_indices:[C, K]
            top_values:[C, K]
        """
        raise NotImplementedError

    # Concepts attribution (Builtin attribution)
    def attribute(
        self,
        batch_X: torch.Tensor,
        target: Union[torch.Tensor | int],
        additional_args: dict = {},
    ) -> torch.Tensor:
        raise NotImplementedError

    # Attention map attribution
    def attribute_attn_map(
        self,
        batch_X: torch.Tensor,
        target: Union[torch.Tensor | int],
    ) -> torch.Tensor:
        """
        Args:
            batch_X: [B, C, W, H]
            target: int/[B, 1]
        Return:
            attribution: [B, 1, _grid, _grid]
        """
        raise NotImplementedError

    @staticmethod
    def attribute_weighted_class(
        concepts_attribution: torch.Tensor,
        concepts_weights: torch.Tensor,
        concepts_idx: torch.Tensor,
        normalized: bool = True,
    ):
        """
        Args:
            concepts_attribution: [B, 1, _grid, _grid];
            concepts_weights: [K, 1];
            concepts_idx: [K, 1];
            normalized: bool;
        Return:
            class_attribution: [1, 1, _grid, _grid];
        """
        if not normalized:
            concepts_weights = concepts_weights / concepts_weights.sum()

        class_attribution = (
            concepts_attribution[concepts_idx] * concepts_weights[:, None, None, None]
        ).sum(dim=0, keepdim=True)

        return class_attribution


# Modified from https://github.com/billpsomas/simpool/blob/master/sp.py
# Original author: Bill Psomas
def show_gradient(grad, parent, name=""):
    if grad.isnan().any():
        print(f"{name} is nan")
        import pdb

        pdb.set_trace()

    return grad


class SimPool(nn.Module):
    SIMPOOL_DEBUG_FLAG = False

    def __init__(
        self,
        dim,
        num_heads=1,
        qkv_bias=False,
        qk_scale=None,
        gamma=None,
        use_beta=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)

        if gamma is not None:
            self.gamma = torch.tensor([gamma], device="cuda")
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0], device="cuda"))
        self.eps = torch.tensor([1e-6], device="cuda")

        self.gamma = gamma
        self.use_beta = use_beta

    def prepare_input(self, x: torch.Tensor):
        if len(x.shape) == 3:  # Transformer
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2)  # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)
            return gap_cls, x
        if len(x.shape) == 4:  # CNN
            # Input tensor dimensions:
            # x: (B, H, W, d), where B is batch size, d is depth (channels), H is height, and W is width
            B, H, W, d = x.shape
            gap_cls = x.mean([1, 2])  # (B, H, W, d) -> (B, d)
            x = x.reshape(B, H * W, d)  # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)
            return gap_cls, x
        else:
            raise ValueError(
                f"Unsupported number of dimensions in input tensor: {len(x.shape)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        prepared_q: torch.Tensor = None,
        return_attn_map: bool = False,
    ):
        if self.SIMPOOL_DEBUG_FLAG:
            import pdb

            pdb.set_trace()
        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        if prepared_q is not None:
            q = prepared_q

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape

        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = (
            self.wq(q)
            .reshape(Bq, Nq, self.num_heads, dq // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = (
            self.wk(k)
            .reshape(Bk, Nk, self.num_heads, dk // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)

        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(
            0, 2, 1, 3
        )  # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)

        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)  # [B, 1, 1, _grid * _grid]

        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(
                attn @ torch.pow((vv - vv.min() + self.eps), self.gamma), 1 / self.gamma
            )  # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)
        if return_attn_map:
            return x.squeeze((1, 2)), attn.squeeze((1, 2))

        return x.squeeze((1, 2))
