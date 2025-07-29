import torch
import torch.nn as nn

from clip.model import CLIP as clip_model_CLIP
from open_clip.model import CLIP as open_clip_model_CLIP

from pcbm.concepts import ConceptBank

from typing import Tuple, Callable, Union, Optional

from utils.model_utils import CBM_Net, CAV, NECLinear

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
        for p in self.CAV_layer.parameters(): p.requires_grad=False
        
        self.num_of_concepts = self.concept_bank.concept_names.__len__()
        self.num_of_classes = num_of_classes
        
        # self.classifier = nn.Linear(self.num_of_concepts, self.num_of_classes)
        self.nec_concepts_projection = NECLinear(self.num_of_concepts, self.num_of_classes, nec=5)
        self.classifier = self.nec_concepts_projection

    # ------------
    # Getter & Setter
    # ------------
    def get_num_classes(self):
        return self.num_of_classes
    
    def get_num_concepts(self):
        return self.num_of_concepts
        
    def train(self, mode=True):
        for p in self.backbone.parameters(): p.requires_grad=not mode
        super().train(mode)
    
    def set_weights(self, weights:torch.Tensor, bias:torch.Tensor):
        self.classifier.weight.data = torch.tensor(weights).to(self.classifier.weight.device)
        self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
        return True

    def state_dict(self):
        return {k:v for k, v in super().state_dict().items() if k.split(".")[0] in self.TRAINABLE_COMPONENTS}
    
    def get_backbone(self) -> open_clip_model_CLIP:
        return self.backbone
    
    def classify(self, 
                 batch_X:torch.Tensor) -> torch.Tensor:
        return self.forward(batch_X=batch_X)
    
    def forward(self, 
                batch_X:torch.Tensor) -> torch.Tensor:
        concept_activations = self.encode_as_concepts(batch_X)

        return self.classifier(concept_activations), concept_activations, None

    def encode_as_embedding(self, 
                        batch_X:torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_X.size()
        
        if self.normalizer is not None:
            batch_X = self.normalizer(batch_X)

        visual_projection = self.backbone.encode_image(batch_X)

        return visual_projection
    
    def encode_as_concepts(self, 
                           batch_X:torch.Tensor) -> torch.Tensor:
        visual_projection = self.encode_as_embedding(batch_X)
        concept_activations = self.CAV_layer(visual_projection)
        return concept_activations
    
    def compute_dist(self, 
                    batch_X:torch.Tensor) -> torch.Tensor:
        return self.encode_as_concepts(batch_X)

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
        weight:torch.Tensor = self.classifier.weight.data  # (out_features, in_features)
        top_values, top_indices = torch.topk(weight, k=K, dim=1)
        top_values = top_values / torch.sum(top_values, dim=1, keepdim=True)
        
        return top_indices, top_values

    def enable_nec(
        self,
        enable:bool,
        nec:int=5
    ) -> None:
        self.nec_concepts_projection.enable_nec = enable
        self.nec_concepts_projection.nec = nec
    
class robust_pcbm(clip_cbm):
    
    TRAINABLE_COMPONENTS = ["backbone"]

    def __init__(self, normalizer, 
                 concept_bank:ConceptBank, 
                 backbone:Union[open_clip_model_CLIP, clip_model_CLIP],
                 num_of_classes:int=10
                 ):
        super().__init__(normalizer, concept_bank, backbone, num_of_classes)
        
    def train(self, mode=True):
        super().train(mode)
        for p in self.backbone.parameters(): p.requires_grad=True
        for p in self.classifier.parameters(): p.requires_grad=False