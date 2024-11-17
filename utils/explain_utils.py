import torch
import torch.nn as nn
import numpy as np

import clip
import open_clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer
from captum.attr import *
from .model_utils import *
from typing import Callable, Union
from scipy.ndimage import gaussian_filter

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
    def saliency_map(posthoc_concept_net:PCBM_Net,):
        saliency = Saliency(posthoc_concept_net)
        return saliency
    
    @staticmethod
    def integrated_gradient(posthoc_concept_net:PCBM_Net,):
        integrated_grad = IntegratedGradients(posthoc_concept_net)
        return integrated_grad
    
    @staticmethod
    def guided_grad_cam(posthoc_concept_net:PCBM_Net):
        raise NotImplementedError
        guided_gradcam = GuidedGradCam(posthoc_concept_net,
                                        getattr(posthoc_concept_net.get_backbone(),
                                                target_layer))
        return guided_gradcam
    
    @staticmethod
    def layer_grad_cam(posthoc_concept_net:PCBM_Net):
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
        elif isinstance(backbone, open_clip.model.CLIP):
             if isinstance(backbone.visual, open_clip.model.ModifiedResNet):
                layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                                getattr(backbone.visual,
                                                        "layer4")[-1])
            
        elif isinstance(backbone, ResNetBottom):
            layer_grad_cam = LayerGradCam(posthoc_concept_net,
                                          backbone.get_submodule("features")
                                            .get_submodule("0")
                                            .get_submodule("stage4")
                                            .get_submodule("unit2")
                                            .get_submodule("body")
                                            .get_submodule("conv2")
                                            .get_submodule("conv"))
        return layer_grad_cam
    
    @staticmethod
    def layer_grad_cam_vit(posthoc_concept_net:PCBM_Net):
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
                            explain_algorithm:Saliency,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target, abs=False)
        return attributions
    
    @staticmethod
    def integrated_gradient(batch_X:torch.Tensor,
                            explain_algorithm:IntegratedGradients,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def guided_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:GuidedGradCam,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        return attributions
    
    @staticmethod
    def layer_grad_cam(batch_X:torch.Tensor,
                            explain_algorithm:LayerGradCam,
                            target:Union[torch.Tensor|int]):
        if isinstance(target, torch.Tensor):
            batch_X = batch_X.expand(target.size(0), -1, -1, -1)

        attributions:torch.Tensor = explain_algorithm.attribute(batch_X, target=target)
        upsampled_attr = LayerAttribution.interpolate(attributions, batch_X.size()[-2:], interpolate_mode="bicubic")
        return upsampled_attr
    
    @staticmethod
    def layer_grad_cam_vit(batch_X:torch.Tensor,
                            explain_algorithm:layer_grad_cam_vit,
                            target:Union[torch.Tensor|int]):
        return __class__.layer_grad_cam(batch_X = batch_X,
                              explain_algorithm = explain_algorithm,
                              target = target)

class attribution_pooling_forward:
    
    @staticmethod
    def max_pooling_class_wise(batch_X:torch.Tensor,
                               attributions:torch.Tensor,
                               concept_idx:Union[torch.Tensor, int],
                               pcbm_net:PCBM_Net):
        if isinstance(concept_idx, int):
            return attributions
        
        max_concept_idx = pcbm_net(batch_X)[0, concept_idx].argmax()
        return attributions[max_concept_idx]
    
    @staticmethod
    def max_pooling_channel_wise(batch_X:torch.Tensor,
                               attributions:torch.Tensor,
                               concept_idx:Union[torch.Tensor, int],
                               pcbm_net:PCBM_Net):
        return attributions.max(0).values
    
    @staticmethod
    def mean_pooling(batch_X:torch.Tensor,
                               attributions:torch.Tensor,
                               concept_idx:Union[torch.Tensor, int],
                               pcbm_net:PCBM_Net):
        return attributions.mean(0)

# Author: Felipe Torres Figueroa - felipe.torres@lis-lab.fr
# Modified from: https://github.com/eclique/RISE/blob/master/evaluation.py
# Functions
def gkern(klen:int, 
          nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

# ========================================================================
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

# ========================================================================
# Causal Metrics
class CausalMetric(nn.Module):
    def __init__(self, model:nn.Module, 
                 mode:str, 
                 step:int,
                 substrate_fn:Callable,
                #  HW:int,
                 classes:int):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): model wrapped in MAE to be explained.
            mode (str): 'del' or 'ins' or 'del-mae'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        super().__init__()
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        # self.HW = HW**2
        self.n_classes = classes

    def forward(self, 
                 img_batch:torch.Tensor,
                 exp_batch:torch.Tensor, 
                 batch_size:int, 
                 top:torch.Tensor):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (torch.Tensor): Array containing scores at every step 
            for every image.
        """
        # Baseline parameters
        B, C, H, W = img_batch.size()
        predictions = torch.FloatTensor(B, self.n_classes)
        
        n_steps = (H * W + self.step - 1) // self.step
        # Flatten -> max to min.
        salient_order = torch.argsort(exp_batch.view(-1, H * W), 
                                           dim=1, 
                                           descending=True)
        r = torch.arange(B).view(B, 1)
        assert B % batch_size == 0
        
        # Forwarding
        #predictions = self.model(img_batch)
        #top = torch.argmax(predictions.detach(), dim=-1).cpu()
        scores = torch.zeros((n_steps + 1, B)).cpu()

        # Generating the starting image-reconstruction to forward
        # substrate = torch.zeros_like(img_batch)
        # for j in range(B // batch_size):
            # Masks every slice of the minibatch using the substrate function
        substrate = self.substrate_fn(img_batch)
            # substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])
            
        if 'del' in self.mode:
            start = img_batch.clone().detach()
            finish = substrate.flatten(start_dim=2)
            
        elif 'ins' in self.mode:
            start = substrate
            finish = img_batch.clone().detach().flatten(start_dim=2)
        with torch.no_grad():
            # While not all pixels are changed
            for i in range(n_steps+1):
                # Iterate over batches
                for j in range(B // batch_size):
                # Iterates over minibatches to retrieve the predicted
                # probabilities for masking step i.
                    # Compute new scores
                    preds = self.model(start[j*batch_size:(j+1)*batch_size])
                    preds = nn.functional.softmax(preds, dim=1)
                    
                    # Gets the predicted probabilities of baseline predictions
                    preds = preds[range(batch_size),
                                top[j*batch_size:(j+1)*batch_size]]
                    
                    # Appends probabilities to scores
                    scores[i, j*batch_size:(j+1)*batch_size] = preds.detach().cpu()
                    
                # Change specified number of most salient px to substrate px
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start = start.flatten(start_dim=2)
                start[r, :, coords] = finish[r, :, coords]
                start = start.view(img_batch.size())
        return scores
