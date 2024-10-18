import torch
import numpy as np
import os
from PIL import Image


def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
    import torchvision
    import matplotlib.pyplot as plt
    
    if comparison_images is not None:
        images = torch.cat((images, comparison_images), dim=3)

    # 使用 torchvision.utils.make_grid 将 64 张图片排列成 8x8 的网格
    grid_img = torchvision.utils.make_grid(images, nrow=2, normalize=True)

    # 转换为 NumPy 格式以便用 matplotlib 显示
    plt.imshow(grid_img.permute(1, 2, 0))  # 转换为 [H, W, C]
    plt.axis('off')  # 隐藏坐标轴
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
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    # pos_mask = attn_map <= 0
    attn_map = normalize(attn_map)
    # attn_map[pos_mask] = 0
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img:np.ndarray, attn_map:np.ndarray, blur=True, prefix:str="", save_to:str=None):
    import matplotlib.pyplot as plt
    attn_map = getAttMap(img, attn_map, blur)

    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(os.path.join(save_to, f"{prefix}-original_image.jpg"))
        
        attn_map_pil = Image.fromarray((attn_map * 255).astype(np.uint8))
        attn_map_pil.save(os.path.join(save_to, f"{prefix}-attn_image.jpg"))
    else:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[1].imshow(attn_map)
        for ax in axes:
            ax.axis("off")
        plt.show()