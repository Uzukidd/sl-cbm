import argparse
import random
import clip.model
import numpy as np
import pickle as pkl
import json
import time
import subprocess
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Callable, Union, Dict

import clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution
from utils import *



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")

    parser.add_argument("--pcbm-arch", default="PCBM", type=str)
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--concept-pooling", default="max_pooling_class_wise", type=str)
    parser.add_argument("--concept-target", required=True, type=str)
    parser.add_argument("--class-target", default="", type=str)
    
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)
    parser.add_argument('--save-100-local', action='store_true')
    parser.add_argument('--zip', action='store_true')


    return parser.parse_args()


class concept_select_func:
    @staticmethod
    def cifar10(model_context: model_pipeline,
                concept_target:str):
        targeted_concept_idx = model_context.concept_bank.concept_names.index(concept_target)
        return targeted_concept_idx
    
    @staticmethod
    def cub(model_context: model_pipeline,
                concept_target:str):
        if hasattr(CUB_features, concept_target):
            # trick to get the device of a nn.Module
            return torch.arange(getattr(CUB_features, concept_target)[0], getattr(CUB_features, concept_target)[1] + 1)\
                .to(next(model_context.posthoc_layer.parameters()).device)
        
        return model_context.concept_bank.concept_names.index(int(concept_target))
    
    @staticmethod
    def rival10(model_context: model_pipeline,
                concept_target:str):
        targeted_concept_idx = model_context.concept_bank.concept_names.index(concept_target)
        return targeted_concept_idx
    
    
def main(args:argparse.Namespace):
    set_random_seed(args.universal_seed)
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    model.eval()

    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, 
                                                    args.explain_method)(forward_func=model.encode_as_concepts,
                                                                        model = model)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    attribution_pooling:Callable[..., torch.Tensor] = getattr(attribution_pooling_forward, args.concept_pooling)
    targeted_concept_idx = getattr(concept_select_func, args.dataset)(model_context, args.concept_target)
    args.logger.info(targeted_concept_idx)
    
    count = 0
    for idx, data in tqdm(enumerate(dataset.test_loader), 
                          total=dataset.test_loader.__len__()):
        batch_X, batch_Y = data
        batch_X:torch.Tensor = batch_X.to(args.device)
        batch_Y:torch.Tensor = batch_Y.to(args.device)
        
        if args.class_target != "" and dataset.idx_to_class[batch_Y.item()] != args.class_target:
            continue
        
        # if posthoc_concept_net.output_as_class(batch_X).item() != batch_Y.item():
        #     continue
        
        batch_X.requires_grad_(True)
        
        attributions:torch.Tensor = explain_algorithm_forward(batch_X=batch_X, 
                                                              explain_algorithm=explain_algorithm,
                                                              target=targeted_concept_idx)
        attributions = attribution_pooling(batch_X = batch_X,
                                           attributions = attributions,
                                           concept_idx = targeted_concept_idx,
                                           pcbm_net = model)
        if args.save_100_local:
            if count == 100:
                break
            save_to = os.path.join(args.save_path, f"{args.explain_method}/{args.concept_target}_images")
            viz_attn(batch_X,
                    attributions,
                    blur=True,
                    prefix=f"{idx:03d}",
                    save_to=save_to)
            try:
                captum_vis_attn(batch_X, 
                                attributions, 
                                title=f"{dataset.idx_to_class[batch_Y.item()]}-attributions: {args.concept_target}",
                                save_to=os.path.join(save_to, f"{idx:03d}-captum-image.jpg"))
            except:
                pass
               
            count += 1

        else:
            for i in range(batch_Y.size(0)):
                print(f"ground truth: {dataset.idx_to_class[batch_Y[i].item()]}")
            topK_concept_to_name(args, model, batch_X)
            viz_attn(batch_X,
                    attributions,
                    blur=True,
                    save_to=None)
            captum_vis_attn(batch_X, 
                        attributions, 
                        title=f"{dataset.idx_to_class[batch_Y.item()]}-attributions: {args.concept_target}",
                        save_to=None)

    # original_Xs = torch.concat(original_Xs, dim = 0)
    # batch_Ys = torch.concat(batch_Ys, dim = 0)
    # adversarial_Xs = torch.concat(adversarial_Xs, dim = 0)
    
    # evaluate_adversarial_sample(ori_adv_pair(
    #     original_X=(adversarial_Xs - original_Xs),
    #     # adversarial_X=adversarial_Xs,
    # ))
    
if __name__ == "__main__":
    args = config()
    args.save_path = os.path.join("./outputs/evals", args.exp_name)
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

    if args.zip:
        command = ["zip", "-r", args.save_path + ".zip", args.save_path]
        subprocess.run(command, check=True)
    