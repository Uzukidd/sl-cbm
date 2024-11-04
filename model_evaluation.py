import argparse
import time

import clip

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *

from utils import *
from asgt import ASGT

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument('--save-100-local', action='store_true')

    return parser.parse_args()
    
def main(args):
    set_random_seed(args.universal_seed)
    concept_bank = load_concept_bank(args)
    backbone, preprocess = load_backbone(args)
    normalizer = transforms.Compose(preprocess.transforms[-1:])
    preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    posthoc_layer = load_pcbm(args)
    trainset, testset, class_to_idx, idx_to_class, train_loader, test_loader = load_dataset(args, preprocess)
    
    model_context = model_pipeline(concept_bank = concept_bank, 
                   posthoc_layer = posthoc_layer, 
                   preprocess = preprocess, 
                   normalizer = normalizer, 
                   backbone = backbone)

    posthoc_concept_net = PCBM_Net(model_context=model_context,
                                   output_logit=True)
    
    asgt_module = ASGT(model = posthoc_concept_net, 
                       training_forward_func = None,
                       loss_func = nn.CrossEntropyLoss(),
                       attak_func="FGSM",
                       explain_func = None,
                       eps = 0.025,
                       k = -1,
                       lam = -1,
                       feature_range= (0.0, 1.0),
                       device=torch.device(args.device))
    
    asgt_module.evaluate_model(train_loader)
    asgt_module.evaluate_model(test_loader)
    robustness = asgt_module.evaluate_model_robustness(test_loader)

    
if __name__ == "__main__":
    args = config()
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    