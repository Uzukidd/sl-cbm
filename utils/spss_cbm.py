import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from clip.model import CLIP as clip_model_CLIP
from open_clip.model import CLIP as open_clip_model_CLIP

from pcbm.concepts import ConceptBank
from pcbm.models import CAV

from typing import Tuple, Callable, Union, Optional

from .model_utils import CBM_Net, SimPool


# SimPooling Semi-Supervised (SPSS) VL-CBM
class spss_pcbm(CBM_Net):

    TRAINABLE_COMPONENTS = ["simpool", 
                            "token_projection",
                            "classifier"]

    def __init__(self, normalizer, 
                 concept_bank:ConceptBank, 
                 backbone:open_clip_model_CLIP,
                 concept_softmax:bool=False,
                 num_of_classes:int=10
                 ):
        super().__init__()

        self.concept_bank = concept_bank
        self.normalizer = normalizer
        self.backbone:open_clip_model_CLIP = backbone
        self.concept_softmax = concept_softmax
        self.embedding_size = self.backbone.visual.output_dim

        assert hasattr(self.backbone.visual, "output_tokens")
        self.backbone.visual.output_tokens = True
        

        self.cavs:torch.Tensor = self.concept_bank.vectors.detach().clone() # [18, D]
        self.concept_names:list[str] = self.concept_bank.concept_names.copy()

        self.n_concepts = self.cavs.shape[0]
        self.num_of_concepts = self.concept_bank.concept_names.__len__()
        self.num_of_classes = num_of_classes

        self.simpool = SimPool(self.num_of_concepts, 
                               num_heads=1, 
                               qkv_bias=False, 
                               qk_scale=None, 
                               use_beta=True)

        self.CAV_layer = CAV(concept_bank.vectors, 
                        concept_bank.intercepts, 
                        concept_bank.norms,
                        concept_bank.concept_names.copy())

        token_width = self.backbone.visual.proj.size(0)
        
        self.token_projection = nn.Linear(in_features=token_width, 
                                        out_features=self.num_of_concepts)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_of_concepts),
            nn.Linear(self.num_of_concepts, self.num_of_classes),
        )

        self._attn_map = None
        for name, param in self.named_parameters(): 
            if name.split(".")[0] in self.TRAINABLE_COMPONENTS:
                param.requires_grad=True
            else:
                param.requires_grad=False

    # ------------
    # Getter & Setter
    # ------------
    def get_num_classes(self):
        return self.num_of_classes
    
    def get_num_concepts(self):
        return self.n_concepts

    def get_expalainable_component(self):
        return self.simpool

    def train(self, mode=True):
        for p in self.backbone.parameters(): p.requires_grad=False
        super().train(mode)

    def set_weights(self, weights:torch.Tensor, bias:torch.Tensor):
        self.classifier[1].weight.data = torch.tensor(weights).to(self.classifier[1].weight.device)
        self.classifier[1].bias.data = torch.tensor(bias).to(self.classifier[1].weight.device)
        return True

    def state_dict(self):
        return {k:v for k, v in super().state_dict().items() if k.split(".")[0] in self.TRAINABLE_COMPONENTS}
    
    def get_backbone(self) -> open_clip_model_CLIP:
        return self.backbone
    
    def forward(self, input_X:torch.Tensor):
        if input_X.size().__len__() == 5:
            B, N, C, W, H = input_X.size()
            images = torch.reshape(input_X, 
                                (B * N, C, W, H))
        else:
            images = input_X
        pooled_tokens, token_concepts = self.encode_as_concepts(images, return_token_concepts=True)
        #     (bs*2,C)         (bs*2,Class)
        return self.classifier(pooled_tokens), pooled_tokens, token_concepts
    
    def encode_as_concepts(self, 
                           batch_X:torch.Tensor,
                           return_token_concepts:bool=False,
                           return_attn_map:bool=False) -> Tuple[torch.Tensor]:
        B, C, H, W = batch_X.size()
        images = self.normalizer(batch_X)

        image_embedding, visual_patches = self.backbone.encode_image(images) # [B, D2]
        # H * W tokens + cls token x D1, cls token x D1 -> proj -> D2 == Embedding 
        # 1x1 convolution
        token_concepts = self.token_projection(visual_patches) # [B, H * W, D1] -> [B, H * W, C]

        # [B, H * W, D] * [C, D]
        if self.concept_softmax:
            token_concepts = F.softmax(token_concepts, dim=2)

        # prepare cavs as query
        # image_embedding @ self.cavs
        cav_query = self.cavs.mean(1).unsqueeze(0).unsqueeze(0).expand((B, -1, -1)) # [C, D2]-> [B, 1, C]
        # cav_query = self.CAV_layer(image_embedding).unsqueeze(1) # [C, D2]-> [B, 1, C]
        # [B, D2] @ [C, D2].T -> [B, C] -> [B, 1, C]
        # cav_query = (image_embedding @ self.cavs.T).unsqueeze(1) # [C, D2]-> [B, 1, C]

        attn_map = None
        if return_attn_map:                     # [B, H * W, C], [B, 1, C]
            pooled_tokens, attn_map = self.simpool(token_concepts, cav_query, return_attn_map=True) # [B, C]
            _grid = int(np.round(np.sqrt(attn_map.size(-1))))
            attn_map = attn_map.view(B, 1, _grid, _grid)
        else:
            pooled_tokens = self.simpool(token_concepts, cav_query) # [B, C]

        res_tuple = (pooled_tokens, )
        if return_token_concepts:
            res_tuple = res_tuple + (token_concepts, )
        
        if return_attn_map:
            res_tuple = res_tuple + (attn_map, )
        
        return res_tuple
    
    def attribute_attn_map(self,
                  batch_X:torch.Tensor) -> torch.Tensor:
        """
            Args: 
                batch_X: [B, C, W, H]
                target: int/[B, 1]
            Return:
                attribution: [B, 1, _grid, _grid]
        """
        # [C] [B, 1, grid, grid]
        _, attn_map = self.encode_as_concepts(batch_X, return_attn_map=True)
        attn_map = F.interpolate(attn_map, batch_X.size()[-2:], mode="bicubic")

        return attn_map
    
    def attribute(self,
                  batch_X:torch.Tensor,
                  target:Union[torch.Tensor|int], 
                  additional_args:dict={}) -> torch.Tensor:
        """
            Args: 
                batch_X: [B, C, W, H]
                target: int/[B, 1]
            Return:
                attribution: [B, 1, _grid, _grid]
        """
        # [C] [B, grid * grid, C]
        _, token_concepts = self.encode_as_concepts(batch_X, return_token_concepts=True)
        B, N, C = token_concepts.size()

        if isinstance(target, torch.Tensor):
            # [B, grid * grid, 1]
            expanded_target = target.unsqueeze(1).unsqueeze(1).expand(-1, N, -1) # [B, N, 1]
            attribution = token_concepts.gather(dim=2, index=expanded_target) # [B, grid * grid, 1]
        else:
            # [1, grid * grid, C]
            attribution = token_concepts[:, :, target:target+1] # [1, grid * grid, 1]
        
        # [B, grid * grid, 1] -> # [B, 1, grid, grid]
        _grid = int(np.round(np.sqrt(N)))
        attribution = attribution.permute((0, 2, 1)).view(B, 1, _grid, _grid)
        return attribution
    

    def get_topK_concepts(self, K:int=5):
        """
            Args:
                K:int
            
            Returns:
                top_indices:[C, K]
                top_values:[C, K]
        """
        weight:torch.Tensor = self.classifier[1].weight.data  # (out_features, in_features)
        top_values, top_indices = torch.topk(weight, k=K, dim=1)
        top_values = top_values / torch.sum(top_values, dim=1, keepdim=True)
        
        return top_indices, top_values

    def forward_projs(self,
                      concept_projs:torch.Tensor) -> torch.Tensor:
        return self.classifier(concept_projs)