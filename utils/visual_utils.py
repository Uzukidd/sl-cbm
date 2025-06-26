import torch
import numpy as np
import os
from captum.attr import visualization
from PIL import Image

from typing import Tuple

def reduce_tensor_as_numpy(input:torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        input: [1, C, W, H] or [C, W, H]
    
    Returns:
        [W, H, C]
    """
    if input.size().__len__() == 4:
        input = input.detach().squeeze(0)
    return input.permute((1, 2, 0)).detach().cpu().numpy()
                
def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
    import torchvision
    import matplotlib.pyplot as plt
    
    if comparison_images is not None:
        images = torch.cat((images, comparison_images), dim=3)

    grid_img = torchvision.utils.make_grid(images, nrow=4, normalize=True)

    plt.imshow(grid_img.permute(1, 2, 0)) 
    plt.axis('off')
    plt.show()
    
def getAttMap(img, attn_map, blur=True):
    import matplotlib.pyplot as plt
    from scipy.ndimage import filters
    def normalize(x: np.ndarray) -> np.ndarray:
        # Normalize to [0, 1].
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(attn_map.shape[:2]))
    # pos_mask = attn_map <= 0
    attn_map = normalize(attn_map)
    # attn_map[pos_mask] = 0
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)

    if img is not None:
        attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
                (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    else:
        attn_map = attn_map_c
    return attn_map

def viz_attn(batch_X:torch.Tensor, attributions:torch.Tensor, blur=True, prefix:str="", save_to:str=None):
    import matplotlib.pyplot as plt
    batch_X = reduce_tensor_as_numpy(batch_X)
    attributions = reduce_tensor_as_numpy(attributions)
    attn_map = getAttMap(batch_X, attributions.sum(2), blur)

    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        img_pil = Image.fromarray((batch_X * 255).astype(np.uint8))
        img_pil.save(os.path.join(save_to, f"{prefix}-original_image.jpg"))
        
        attn_map_pil = Image.fromarray((attn_map * 255).astype(np.uint8))
        attn_map_pil.save(os.path.join(save_to, f"{prefix}-attn_image.jpg"))
    else:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(batch_X)
        axes[1].imshow(attn_map)
        for ax in axes:
            ax.axis("off")
        plt.show()
        
def viz_attn_only(batch_X:torch.Tensor, attributions:torch.Tensor, blur=True, prefix:str="", save_to:str=None):
    import matplotlib.pyplot as plt
    batch_X = reduce_tensor_as_numpy(batch_X)
    attributions = reduce_tensor_as_numpy(attributions)
    
    attn_map = None
    attn_map = getAttMap(batch_X, attributions.sum(2), blur)

    plt.imshow(np.clip(attn_map, 0.0, 1.0))
    plt.axis("off")
    plt.tight_layout()
    
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(os.path.join(save_to, f"{prefix}-attn_image.jpg"), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
        
def viz_attn_multiple(batch_X:torch.Tensor, attributions:list[torch.Tensor], blur=True, prefix:str="", save_to:str=None):
    import matplotlib.pyplot as plt
    batch_X = reduce_tensor_as_numpy(batch_X)
    attributions = [reduce_tensor_as_numpy(attribution) for attribution in attributions]
    
    attn_map = []
    for attribution in attributions:
        attn_map.append(getAttMap(batch_X, attribution.sum(2), blur))

    
    _, axes = plt.subplots(1, 1 + attn_map.__len__(), figsize=(10, 5))
    axes[0].imshow(np.clip(batch_X, 0.0, 1.0))
    for idx, map in  enumerate(attn_map):
        axes[1 + idx].imshow(np.clip(map, 0.0, 1.0))
    
    for ax in axes:
        ax.axis("off")
    
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(os.path.join(save_to, f"{prefix}-attn_image.jpg"), bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()
        
def captum_vis_attn(batch_X:torch.Tensor, attributions:torch.Tensor, title:str=None, save_to:str=None):
    batch_X = reduce_tensor_as_numpy(batch_X)
    attributions = reduce_tensor_as_numpy(attributions)
    figure, axis = visualization.visualize_image_attr_multiple(attributions, 
                                    batch_X, 
                                    signs=["all", 
                                        "positive",
                                        "positive",
                                        "positive",
                                        "positive"],
                                    titles=[None,
                                            None,
                                            title,
                                            None,
                                            None],
                                    use_pyplot=save_to is None,
                                    methods=["original_image", "heat_map", "blended_heat_map", "masked_image", "alpha_scaling"],)
    if save_to is not None:
        figure.savefig(save_to, format='jpg', dpi=300)