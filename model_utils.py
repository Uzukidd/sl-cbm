import torch
import torch.nn as nn
from clip.model import CLIP
from common_utils import model_pipeline

class PCBM_Net(nn.Module):
    def __init__(self, model_context:model_pipeline):
        super().__init__()
        self.normalizer = model_context.normalizer
        self.backbone = model_context.backbone
        self.posthoc_layer = model_context.posthoc_layer
        
    def forward(self, input_x:torch.Tensor, output_class:bool=False):
        batch_X_normalized = self.normalizer(input_x)
        embeddings = self.backbone.encode_image(batch_X_normalized)
        concept_projs = self.posthoc_layer.compute_dist(embeddings)
        
        if output_class:
           return self.posthoc_layer.forward_projs(concept_projs).argmax(1)
        
        return concept_projs
    
    def get_backbone(self):
        return self.backbone