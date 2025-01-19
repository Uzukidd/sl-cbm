import argparse
import random
import clip.model
import numpy as np
import pickle as pkl
import json
import time
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Callable, Union, Dict

import clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer
from open_clip_train.train import train_one_epoch, evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms

from autoattack import AutoAttack

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution
from utils import *
from asgt import robust_training, ASGT_Legacy


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    
    # PCBM, classifier, cub_net, etc
    parser.add_argument("--pcbm-arch", default="PCBM", type=str)
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument("--train-method", required=True, type=str)
    parser.add_argument("--train-embedding", action='store_true')
    parser.add_argument('--not-save-ckpt', action='store_true')

    
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--eps", default=0.025, type=float)
    parser.add_argument("--k", default=1e-1, type=float)
    
    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)

    return parser.parse_args()

# main_cocnept_classes = {
#     "airplane" : "propellers",
#     "automobile" : "windshield",
#     "bird" : "beak",
#     "cat" : "sharp claws",
#     "deer" : "antler",
#     "dog" : "paws",
#     "frog" : "amphibian",
#     "horse" : "horseback",
#     "ship" : "porthole",
#     "truck" : "an engine"
# }

# def embedding_robust_loss_func(context:robust_training,
#                                 batch_X:torch.Tensor,
#                                 batch_Y:torch.Tensor):
#     import pdb; pdb.set_trace()
#     context.model.eval()
#     embedding_Y = context.model(batch_X)
#     batch_adv_X = context.generate_adv_sample(batch_X, embedding_Y)
#     # masked_batch_adv_X = self.generate_masked_sample(batch_adv_X, batch_Y)
    
#     # self.model.train()
#     # clean_logit = self.model(batch_X)
#     # adv_logit = self.model(batch_adv_X)
#     # masked_adv_logit = self.model(masked_batch_adv_X)
    
#     # clean_prob_log = nn.functional.log_softmax(clean_logit, dim=1)
#     # masked_adv_prob_log = nn.functional.log_softmax(masked_adv_logit, dim=1)
    
#     # loss = self.loss_func(clean_logit, batch_Y) \
#     #         + self.loss_func(adv_logit, batch_Y) \
#     #         + self.lam * nn.functional.kl_div(clean_prob_log, 
#     #                                             masked_adv_prob_log, 
#     #                                             reduction='batchmean', 
#     #                                             log_target=True)
#     # embedding_loss = 
    
def training_forward_func(loss:torch.Tensor, 
                          model:nn.Module, 
                          optimizer:optim.Optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# class prepare_classification_model:
    
#     @staticmethod
#     def clip(args:argparse.Namespace,
#               posthoc_concept_net:PCBM_Net):
#         posthoc_concept_net.posthoc_layer.classifier.weight.requires_grad_(False)
#         posthoc_concept_net.posthoc_layer.classifier.bias.requires_grad_(False)

#         return posthoc_concept_net
    
#     @staticmethod
#     def open_clip(args:argparse.Namespace,
#               posthoc_concept_net:PCBM_Net):
#         return __class__.clip(args, 
#                               posthoc_concept_net)
        
#     @staticmethod
#     def resnet18_cub(args:argparse.Namespace,
#               posthoc_concept_net:PCBM_Net):
#         return posthoc_concept_net.backbone
    
# class save_checkpoint:
    
#     @staticmethod
#     def clip(args:argparse.Namespace, 
#              model:PCBM_Net):
#         torch.save({"state_dict": model.backbone.state_dict()}, 
#                    os.path.join(args.save_path, f"{args.train_method}-{args.backbone_name.replace("/", "-")}.pth"))
        
#     @staticmethod
#     def open_clip(args:argparse.Namespace, 
#                   model:PCBM_Net):
#         return __class__.clip(args, 
#                               model)
    
#     @staticmethod
#     def resnet18_cub(args:argparse.Namespace, 
#                      model:nn.Module):
#         return torch.save({"state_dict": model.state_dict()}, 
#                           os.path.join(args.save_path, f"{args.train_method}-{args.backbone_name.replace("/", "-")}.pth"))

def main(args:argparse.Namespace):
    set_random_seed(args.universal_seed)

    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    args.data_size = dataset.trainset[0][0].size()
    args.logger.info(f"Trainable components: {model.TRAINABLE_COMPONENTS}")
    
    # Get explain algorithm
    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, args.explain_method)(forward_func=model.classify, model = model)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    
    # Prepare training module
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    
    args.logger.info(f"data size: {args.data_size}")
    asgt_module = robust_training(model = model, 
                        model_forward_func=model.classify,
                       training_forward_func = partial(training_forward_func,
                                                       model = model,
                                                       optimizer = optimizer),
                       loss_func = loss_func,
                       attak_func="FGSM",
                       explain_func = partial(explain_algorithm_forward, 
                                              explain_algorithm=explain_algorithm),
                       robust_loss_func=args.train_method,
                       eps = args.eps,
                       k = int(args.data_size[-1] * args.data_size[-2] * args.k),
                       lam = 1.0,
                       feature_range= (0.0, 1.0),
                       device=torch.device(args.device))
    
    # asgt_module = ASGT_Legacy(model = model, 
    #                    training_forward_func = partial(training_forward_func,
    #                                                    model = model,
    #                                                    optimizer = optimizer),
    #                    loss_func = nn.CrossEntropyLoss(),
    #                    attak_func="FGSM",
    #                    explain_func = partial(explain_algorithm_forward, 
    #                                           explain_algorithm=explain_algorithm),
    #                    eps = args.eps,
    #                    k = int(args.data_size[-1] * args.data_size[-2] * args.k),
    #                    lam = 1.0,
    #                    feature_range= (0.0, 1.0),
    #                    device=torch.device(args.device))
    
    totall_accuracy = asgt_module.evaluate_model(dataset.train_loader)
    args.logger.info(f"Accuracy: {100 * totall_accuracy:.2f}%")

    totall_accuracy = asgt_module.evaluate_model(dataset.test_loader)
    args.logger.info(f"Accuracy: {100 * totall_accuracy:.2f}%")

    totall_accuracy = asgt_module.evaluate_model_robustness(dataset.test_loader)
    args.logger.info(f"Robustness accuracy: {100 * totall_accuracy:.2f}%")
        
    num_epoches = 10
    for epoch in range(num_epoches):
        running_loss = asgt_module.train_one_epoch(dataset.train_loader)
        args.logger.info(f"Epoch [{epoch + 1}/{num_epoches}], Loss: {running_loss / len(dataset.train_loader):.4f}")
        
        if not args.not_save_ckpt:
            save_to = os.path.join(args.save_path, f"{args.pcbm_arch}_{args.backbone_name}.pt")
            torch.save({"state_dict": model.backbone.state_dict()}, save_to)
        
            totall_accuracy = asgt_module.evaluate_model(dataset.train_loader)
            args.logger.info(f"Accuracy: {100 * totall_accuracy:.2f}%")

            totall_accuracy = asgt_module.evaluate_model(dataset.test_loader)
            args.logger.info(f"Accuracy: {100 * totall_accuracy:.2f}%")

            totall_accuracy = asgt_module.evaluate_model_robustness(dataset.test_loader)
            args.logger.info(f"Robustness accuracy: {100 * totall_accuracy:.2f}%")
    
    
if __name__ == "__main__":
    args = config()
    args.save_path = os.path.join("./outputs", args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=4)
    
    args.logger = common_utils.create_logger(log_file = os.path.join(args.save_path, "exp_log.log"))
    args.logger.info(args_json)
    args.logger.info(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.logger.info(f"GPU devices failed. Change to {args.device}")
    main(args)
    