import torch
import torch.nn as nn
import numpy as np

import clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer
from captum.attr import GradientAttribution, LayerAttribution, Saliency
from .model_utils import *
from typing import Callable, Union

class layer_grad_cam_vit:
    def __init__(self, forward_func:Callable, target_layer:nn.Module) -> None:
        self.forward_func = forward_func
        self.target_layer = target_layer
        self.gradients:torch.Tensor = None
        self.activations:torch.Tensor = None
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation))
        self.handles.append(
            target_layer.register_forward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        activation:torch.Tensor = output
        self.activations = activation.detach().permute((1, 0, 2))[:, 1:, :]

    def save_gradient(self, module, input, output:torch.Tensor):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        
        def _store_grad(grad:torch.Tensor):
            self.gradients = grad.detach().permute((1, 0, 2))[:, 1:, :]

        output.register_hook(_store_grad)
    
    def attribute(self, batch_X:torch.Tensor, target:Union[torch.Tensor|int], additional_args:dict={}):
        self.gradients:torch.Tensor = None # [B, grid ** 2, F]
        self.activations:torch.Tensor = None # [B, grid ** 2, F]
        
        output = self.forward_func(batch_X, **additional_args)
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]
            
        self.forward_func.zero_grad()
        loss = output[:, target]
        loss.backward()
        
        B, F = self.gradients.size(0), self.gradients.size(2)
        
        importance_weights = self.gradients.mean(dim=(1)) # [B, F]
        
        weighted_activations = importance_weights[:, None, :] * self.activations # [B, grid ** 2, F]
        weighted_activations = weighted_activations.permute(0, 2, 1) # [B, F, grid ** 2]
        _grid = int(np.round(np.sqrt(weighted_activations.size(2))))
        weighted_activations = weighted_activations.reshape(B, F, _grid, _grid) # [B, F, grid, grid]

        return weighted_activations
    
class model_explain_algorithm_factory:
    
    @staticmethod
    def saliency_map(args, 
                    posthoc_concept_net:PCBM_Net,):
        from captum.attr import IntegratedGradients
        saliency = Saliency(posthoc_concept_net)
        return saliency
    
    @staticmethod
    def integrated_gradient(args, 
                            posthoc_concept_net:PCBM_Net,):
        from captum.attr import IntegratedGradients
        integrated_grad = IntegratedGradients(posthoc_concept_net)
        return integrated_grad
    
    @staticmethod
    def guided_grad_cam(args, 
                        posthoc_concept_net:PCBM_Net):
        from captum.attr import GuidedGradCam
        raise NotImplementedError
        guided_gradcam = GuidedGradCam(posthoc_concept_net,
                                        getattr(posthoc_concept_net.get_backbone(),
                                                target_layer))
        return guided_gradcam
    
    @staticmethod
    def layer_grad_cam(args, 
                posthoc_concept_net:PCBM_Net):
        from captum.attr import LayerGradCam
        layer_grad_cam = None
        backbone = posthoc_concept_net.backbone
        if isinstance(backbone, CLIP):
            if isinstance(backbone.visual, ModifiedResNet):
                layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                                getattr(backbone.visual,
                                                        "layer4")[-1])
            
            # elif isinstance(backbone.visual, VisionTransformer):
            #     image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
            #     # print(image_attn_blocks[0].attn_probs.size())
            #     # image_attn_blocks = backbone.visual.transformer
            #     # image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
            #     # print(image_attn_blocks)
            #     last_blocks = image_attn_blocks[-2]
            #     # print(last_blocks)
            #     layer_grad_cam = LayerGradCam(posthoc_concept_net,
            #                                     last_blocks)
            #     # layer_grad_cam = layer_grad_cam_vit(posthoc_concept_net,
            #     #                                 last_blocks)
            
        elif isinstance(backbone, ResNetBottom):
            layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                          backbone.get_submodule("features").get_submodule("0").get_submodule("stage4") )
            
        return layer_grad_cam
    
    @staticmethod
    def layer_grad_cam_vit(args, 
                posthoc_concept_net:PCBM_Net):
        from captum.attr import LayerGradCam
        layer_grad_cam = None
        backbone = posthoc_concept_net.backbone
        if isinstance(backbone, CLIP) and isinstance(backbone.visual, VisionTransformer):
            image_attn_blocks = list(dict(backbone.visual.transformer.resblocks.named_children()).values())
            last_blocks = image_attn_blocks[-1].ln_1
            layer_grad_cam = layer_grad_cam_vit(posthoc_concept_net,
                                            last_blocks)
        else:
            raise NotImplementedError
                    
        return layer_grad_cam
    
class model_explain_algorithm_forward:
    
    @staticmethod
    def saliency_map(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def integrated_gradient(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def guided_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def layer_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        upsampled_attr = LayerAttribution.interpolate(attributions, batch_X.size()[-2:], interpolate_mode="bicubic")
        return upsampled_attr
    
    @staticmethod
    def layer_grad_cam_vit(batch_X:torch.Tensor,
                            explain_algorithm:GradientAttribution,
                            target:Union[torch.Tensor|int]):
        return __class__.layer_grad_cam(batch_X = batch_X,
                              explain_algorithm = explain_algorithm,
                              target = target)