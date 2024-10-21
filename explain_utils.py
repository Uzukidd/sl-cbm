import torch
import torch.nn as nn
import numpy as np
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
    
    def attribute(self, batch_X:torch.Tensor, target:Union[torch.Tensor|int]):
        self.gradients:torch.Tensor = None # [B, grid ** 2, F]
        self.activations:torch.Tensor = None # [B, grid ** 2, F]
        
        output = self.forward_func(batch_X)
        self.forward_func.zero_grad()
        loss = output[:, target]
        loss.backward()
        
        B, F = self.gradients.size(0), self.gradients.size(2)
        
        importance_weights = self.gradients.mean(dim=(1)) # [B, F]
        
        weighted_activations = importance_weights[:, None, :] * self.activations # [B, grid ** 2, F]
        weighted_activations = weighted_activations.permute(0, 2, 1) # [B, F, grid ** 2]
        _grid = int(np.round(np.sqrt(weighted_activations.size(2))))
        weighted_activations = weighted_activations.reshape(B, F, _grid, _grid)

        return weighted_activations