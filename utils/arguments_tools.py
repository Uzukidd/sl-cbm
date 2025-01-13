import torch

from tap import Tap

from typing import Literal, Union, Tuple, List

class backbone_configure(Tap):
    backbone_name:str
    backbone_ckpt:str
    device:torch.device

class concept_bank_configure(Tap):
    concept_bank:str
    device:torch.device