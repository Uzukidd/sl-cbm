import argparse
import time

import heapq
import clip
import pickle as pkl
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from rival10.constants import RIVAL10_constants

from pcbm.learn_concepts_multimodal import *

from utils import *

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    parser.add_argument("--concept-bank", default="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10_CSSCBM.pkl", type=str, help="Path to the concept bank")
    
    parser.add_argument("--pcbm-arch", required=True, type=str, help="Bottleneck model architecture")
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")

    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", required=True, type=str)

    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--concept-pooling", default="max_pooling_class_wise", type=str)
    
    parser.add_argument("--target-dataset", default="rival10_full", type=str)
    parser.add_argument("--dataset", default="rival10_full", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument('--dataset-scalar', default=None, type=float)


    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)

    return parser.parse_args()

            
def main(args):
    set_random_seed(args.universal_seed)
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    if "spss" in args.pcbm_arch:
        model = spss_pcbm(
            model_context.normalizer,
            model_context.concept_bank,
            model_context.backbone,
            "simpool",
            False,
            dataset.idx_to_class.__len__(),
        )
        if args.pcbm_ckpt is not None:
            assert os.path.exists(args.pcbm_ckpt) and os.path.isfile(args.pcbm_ckpt)
            model.load_state_dict(torch.load(args.pcbm_ckpt), strict=False)
            print(f"Successfully loaded checkpoint from {args.pcbm_ckpt}")
        model.to(args.device)
    model.eval()
    # dataset.test_loader = DataLoader(
    #         dataset.testset,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         num_workers=args.num_workers,
    #     )
    res = image_generate(args, model, backbone.preprocess, dataset, concept_bank, args.explain_method, 100)
    
    

if __name__ == "__main__":
    args = config()
    args.save_path = os.path.join("./outputs/image_visualization", args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=4)
    
    args.logger = common_utils.create_logger(log_file = os.path.join(args.save_path, "exp_log.log"))
    args.logger.info(args_json)
    args.logger.info(f"universal seed: {args.universal_seed}")

    if not torch.cuda.is_available():
        args.device = "cpu"
        args.logger.info(f"GPU devices failed. Change to {args.device}")
    main(args)
