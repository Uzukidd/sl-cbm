import argparse
import os
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
    parser.add_argument("--pcbm-ckpt", type=str)
    parser.add_argument("--pcbm-arch", default="css_pcbm", type=str)

    parser.add_argument("--dataset", default="css_rival10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument('--evaluate', action='store_true')
    
    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)

    return parser.parse_args()

def estimate_top_concepts_accuracy(concept_predictions, concept_labels):

    bs = concept_predictions.shape[0]
    mini_batch_correct_concepts = 0
    mini_batch_total_concepts = 0

    for i in range(bs):

        k_val = int(torch.sum(concept_labels[i]).item())  # active labels for that class
        _, top_gt_indices = torch.topk(concept_labels[i], k=k_val, dim=-1)
        _, top_pred_indices = torch.topk(concept_predictions[i], k=k_val, dim=-1)

        for k in top_pred_indices:
            mini_batch_total_concepts+=1
            if k in top_gt_indices: mini_batch_correct_concepts+=1

    return mini_batch_correct_concepts, mini_batch_total_concepts

def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    contrastive_loss = []
    classifier_loss = []
    concept_loss = []

    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0
    
    model.train()

    ###Iterating over data loader
    for i, (images, class_labels, concept_labels, use_concept_labels) in enumerate(train_data_loader):
        
        #Loading data and labels to device
        images = images.squeeze().to(device)
        class_labels = class_labels.squeeze().to(device)
        concept_labels = concept_labels.squeeze().to(device)
        use_concept_labels = use_concept_labels.squeeze().to(device)

        # stack the labels along batch dimension (no longer need to be in pairs)
        class_labels = torch.reshape(class_labels,(class_labels.shape[0]*2,1)).squeeze()
        concept_labels = torch.reshape(concept_labels,(concept_labels.shape[0]*2,18))
        use_concept_labels = torch.reshape(use_concept_labels,(use_concept_labels.shape[0]*2,1)).squeeze()

        #Reseting Gradients
        optimizer.zero_grad()

        #Forward
        concept_predictions, class_predictions = model(images)

        #Calculating Loss
        loss1, loss2, loss3 = loss_fn(concept_predictions, class_predictions, class_labels, concept_labels, use_concept_labels)
        _loss = loss1 + loss2 + loss3

        contrastive_loss.append(loss1.item())
        classifier_loss.append(loss2.item())
        concept_loss.append(loss3.item())

        #Backward
        _loss.backward()
        optimizer.step()

        if i%200 == 0:
            print("Contrastive Loss = ",np.mean(contrastive_loss), ", Classifier Loss = ",np.mean(classifier_loss),", Concept Loss = ", np.mean(concept_loss))
        
        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
        total_samples += len(class_labels)
        mbcc, mbtc = estimate_top_concepts_accuracy(concept_predictions, concept_labels)
        concept_acc += mbcc
        concept_count += mbtc


    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    return acc, total_concept_acc


def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    contrastive_loss = []
    classifier_loss = []
    concept_loss = []

    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (images, class_labels, concept_labels, use_concept_labels) in enumerate(val_data_loader):
            
            #Loading data and labels to device
            images = images.squeeze().to(device)
            class_labels = class_labels.squeeze().to(device)
            concept_labels = concept_labels.squeeze().to(device)
            use_concept_labels = use_concept_labels.squeeze().to(device)

            # stack the labels along batch dimension (no longer need to be in pairs)
            class_labels = torch.reshape(class_labels,(class_labels.shape[0]*2,1)).squeeze()
            concept_labels = torch.reshape(concept_labels,(concept_labels.shape[0]*2,18))
            use_concept_labels = torch.reshape(use_concept_labels,(use_concept_labels.shape[0]*2,1)).squeeze()

            #Forward
            concept_predictions, class_predictions = model(images)

            #Calculating Loss
            loss1, loss2, loss3 = loss_fn(concept_predictions, class_predictions, class_labels, concept_labels, use_concept_labels)

            contrastive_loss.append(loss1.item())
            classifier_loss.append(loss2.item())
            concept_loss.append(loss3.item())
            
            # calculate acc per minibatch
            sum_correct_pred += (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
            total_samples += len(class_labels)
            mbcc, mbtc = estimate_top_concepts_accuracy(concept_predictions, concept_labels)
            concept_acc += mbcc
            concept_count += mbtc

    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    return acc, total_concept_acc

def main(args:argparse.Namespace):
    set_random_seed(args.universal_seed)
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)

    # Prepare training module
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = trinity_loss()
    
    # args.logger.info(f"data size: {args.data_size}")
    args.logger.info("\n\n\n\t Model Loaded")
    args.logger.info("\t Total Params = %d",sum(p.numel() for p in model.parameters()))
    args.logger.info("\t Trainable Params = %d",sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.evaluate:
        val_acc, val_concept_acc = val_one_epoch(dataset.test_loader, model, loss_func, args.device)
        args.logger.info("\t Val Class Accuracy = {} and Val Concept Accuracy = {}.".format(round(val_acc,2),round(val_concept_acc,2)))
        return

    for epoch in range(5):
        begin = time.time()

        ###Training
        acc, concept_acc = train_one_epoch(dataset.train_loader, model, optimizer, loss_func, args.device)
        ###Validation
        val_acc, val_concept_acc = val_one_epoch(dataset.test_loader, model, loss_func, args.device)

        save_to = os.path.join(args.save_path, f"css_cbm_{args.backbone_name}.pt")
        torch.save(model.state_dict(), save_to)

        args.logger.info('\n\t Epoch.... %d', epoch + 1)
        args.logger.info("\t Train Class Accuracy = {} and Train Concept Accuracy = {}.".format(round(acc,2),round(concept_acc,2)))
        args.logger.info("\t Val Class Accuracy = {} and Val Concept Accuracy = {}.".format(round(val_acc,2),round(val_concept_acc,2)))
        args.logger.info('\t Time per epoch (in mins) = %d %s', round((time.time()-begin)/60,2),'\n\n')
    
    
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
        args.device = "cpu"
        args.logger.info(f"GPU devices failed. Change to {args.device}")
    main(args)
    