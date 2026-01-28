import torch

# from tap import Tap
import argparse
import time
import datetime
from typing import Literal, Union, Tuple, List

# class backbone_configure(Tap):
#     backbone_name:str
#     backbone_ckpt:str
#     device:torch.device

# class concept_bank_configure(Tap):
#     concept_bank:str
#     device:torch.device


def parse_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--universal-seed",
        default=int(time.time()),
        type=int,
        help="Universal random seed",
    )

    parser.add_argument(
        "--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt"
    )
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)

    parser.add_argument(
        "--concept-bank", required=True, type=str, help="Path to the concept bank"
    )

    # PCBM, classifier, cub_net, etc
    parser.add_argument("--pcbm-ckpt", type=str)
    parser.add_argument("--pcbm-arch", default="spss_pcbm", type=str)

    parser.add_argument("--dataset", default="spss_rival10", type=str)
    parser.add_argument("--target-dataset", default="rival10_full", type=str)

    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    parser.add_argument("--loss", default="spss", type=str)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--use-concept-softmax", action="store_true")

    parser.add_argument("--lambda1", default=1.0, type=float)
    parser.add_argument("--lambda2", default=1.0, type=float)
    parser.add_argument("--lambda3", default=5.0, type=float)
    parser.add_argument("--lambda4", default=1.0, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)

    parser.add_argument("--explain-method", type=str)

    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--intervention", action="store_true")

    parser.add_argument("--dataset-scalar", default=None, type=float)
    parser.add_argument("--not-save-ckpt", action="store_true")
    parser.add_argument("--batch-vis", action="store_true")

    parser.add_argument(
        "--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str
    )
