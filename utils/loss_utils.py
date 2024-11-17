import torch
import torch.nn as nn


class FARE_loss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input:torch.Tensor, 
                    target:torch.Tensor):
        pass
        
    