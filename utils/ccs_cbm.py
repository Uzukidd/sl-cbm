import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.model import CLIP as clip_model_CLIP
from open_clip.model import CLIP as open_clip_model_CLIP

from pcbm.concepts import ConceptBank
from pcbm.models import CAV

from typing import Tuple, Callable, Union, Optional

from .model_utils import CBM_Net

# Contrastive Semi-Supervised (CSS) VL-CBM
class css_pcbm(CBM_Net):

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

    # ------------
    # Getter & Setter
    # ------------
    def get_num_classes(self):
        return self.num_of_classes
    
    def get_num_concepts(self):
        return self.num_of_concepts

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
        if image_pairs.shape.__len__() == 5:
            bs, imgs, channels, h, w = image_pairs.shape
            images = torch.reshape(image_pairs, 
                                (bs*imgs, channels, h, w))
        else:
            images = image_pairs
        images = self.normalizer(images)

        visual_projection, visual_patches = self.backbone.encode_image(images)
        concept_projections = self.concept_projection(torch.mean(visual_patches, 
                                                                 dim=1))

        concept_activations = self.CAV_layer(F.normalize(visual_projection, 
                                                         dim=-1))
        concepts = concept_activations + concept_projections
        #     (bs*2,18)         (bs*2,10)
        return self.classifier(concepts), F.sigmoid(concepts), None
    
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

