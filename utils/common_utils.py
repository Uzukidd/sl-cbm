import argparse
import random
import numpy as np
import pickle as pkl
import json
import logging

from tqdm import tqdm
from typing import Tuple, Callable, Union, Optional, Dict
from dataclasses import dataclass, field
from functools import partial


import clip
from clip.model import CLIP as clip_model_CLIP
import open_clip
from open_clip.model import CLIP as open_clip_model_CLIP

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data  import DataLoader, Dataset, Sampler
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, PosthocHybridCBM, get_model
from pcbm.training_tools import load_or_compute_projections

from .constants import dataset_constants, RIVAL10_features
from .model_utils import *


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class dataset_configure:
    dataset:str
    batch_size:int
    num_workers:int
    
@dataclass
class dataset_collection:
    trainset:Dataset
    testset:Dataset
    class_to_idx:Dict[str ,int]
    idx_to_class:Dict[int ,str]
    train_loader:DataLoader
    test_loader:DataLoader


class class_specific_sampler(Sampler):
    def __init__(self, dataset:Dataset, target_class:int):
        self.dataset = dataset
        self.target_class = target_class
        self.indices = np.where(np.array(dataset.targets) == target_class)[0]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def load_dataset(args:Union[argparse.Namespace, dataset_configure], 
                 preprocess:transforms.Compose, target_class:Optional[int]=None):
    trainset, testset = None, None
    if isinstance(args, argparse.Namespace):
        args = dataset_configure(
            dataset = args.dataset,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
        )
    
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=dataset_constants.CIFAR10_DIR, train=True,
                                    download=False, transform=preprocess)
        testset = datasets.CIFAR10(root=dataset_constants.CIFAR10_DIR, train=False,
                                    download=False, transform=preprocess)
        classes = trainset.classes
        
        train_sampler = None
        test_sampler = None
        if target_class is not None:
            train_sampler = class_specific_sampler(trainset, target_class)
            test_sampler = class_specific_sampler(testset, target_class)
        
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                    sampler = train_sampler,
                                    shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size,
                                    sampler = test_sampler,
                                    shuffle=False, num_workers=args.num_workers)
    
    
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=dataset_constants.CIFAR100_DIR, train=True,
                                    download=False, transform=preprocess)
        testset = datasets.CIFAR100(root=dataset_constants.CIFAR100_DIR, train=False,
                                    download=False, transform=preprocess)
        classes = trainset.classes
        
        train_sampler = None
        test_sampler = None
        if target_class is not None:
            train_sampler = class_specific_sampler(trainset, target_class)
            test_sampler = class_specific_sampler(testset, target_class)
        
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                    sampler = train_sampler,
                                    shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size,
                                    sampler = test_sampler,
                                    shuffle=False, num_workers=args.num_workers)


    elif args.dataset == "cub":
        from pcbm.data.cub import load_cub_data
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(dataset_constants.CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(dataset_constants.CUB_PROCESSED_DIR, "test.pkl")
        if target_class is not None:
            raise NotImplementedError
        
        trainset, train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=dataset_constants.CUB_DATA_DIR, resol=224, normalizer=None,
            n_classes=num_classes, resampling=True)

        testset, test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=dataset_constants.CUB_DATA_DIR, resol=224, normalizer=None,
                n_classes=num_classes, resampling=True)

        classes = open(os.path.join(dataset_constants.CUB_DATA_DIR, "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        print(len(classes), "num classes for cub")
        print(len(train_loader.dataset), "training set size")
        print(len(test_loader.dataset), "test set size")
        

    elif args.dataset == "ham10000":
        raise NotImplementedError
        from pcbm.data.derma_data import load_ham_data
        train_loader, test_loader, idx_to_class = load_ham_data(args, preprocess)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    elif args.dataset == "rival10":
        from utils import LocalRIVAL10
        trainset = LocalRIVAL10(train=True, classification_output=True, masks_dict=False, transform=preprocess)
        testset = LocalRIVAL10(train=False, classification_output=True, masks_dict=False, transform=preprocess)

        class_to_idx = {c: i for (i,c) in enumerate(RIVAL10_features._ALL_CLASSNAMES)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
        
    elif args.dataset == "rival10_full":
        from utils import LocalRIVAL10
        trainset = LocalRIVAL10(train=True, classification_output=False, masks_dict=True, transform=preprocess)
        testset = LocalRIVAL10(train=False, classification_output=False, masks_dict=True, transform=preprocess)

        class_to_idx = {c: i for (i,c) in enumerate(RIVAL10_features._ALL_CLASSNAMES)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
        
    elif args.dataset == "css_rival10":
        from utils import CSS_Rival_Dataset
        train_dataset = CSS_Rival_Dataset(split="train", true_batch_size=args.batch_size, 
                                          percentage_of_concept_labels_for_training=0.01, 
                                          transform=preprocess)
        test_dataset = CSS_Rival_Dataset(split="test", true_batch_size=args.batch_size, 
                                         percentage_of_concept_labels_for_training=0.0, 
                                         transform=preprocess)

        class_to_idx = {c: i for (i,c) in enumerate(RIVAL10_features._ALL_CLASSNAMES)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
    else:
        raise ValueError(args.dataset)
    
    return dataset_collection(
        trainset = trainset, 
        testset = testset, 
        class_to_idx = class_to_idx, 
        idx_to_class = idx_to_class, 
        train_loader = train_loader, 
        test_loader = test_loader
    )
     

@dataclass
class concept_bank_configure:
    concept_bank:str
    device:Union[str, torch.device]

def load_concept_bank(args:Union[argparse.Namespace, concept_bank_configure]) -> ConceptBank:
    
    if isinstance(args, argparse.Namespace):
        args = concept_bank_configure(
            concept_bank = args.concept_bank,
            device = args.device
        )
    
    all_concepts = pkl.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)
    
    return concept_bank


@dataclass
class backbone_configure:
    backbone_name:str
    backbone_ckpt:str
    device:Union[str, torch.device]

@dataclass
class backbone_pipeline:
    preprocess:transforms.Compose
    normalizer:transforms.Compose
    backbone_model:Union[nn.Module, clip_model_CLIP, open_clip_model_CLIP]
    additional_components:dict=field(default_factory=dict)

def load_backbone(args:Union[argparse.Namespace, backbone_configure]) -> backbone_pipeline:
        
    if isinstance(args, argparse.Namespace):
        args = backbone_configure(
            backbone_name = args.backbone_name,
            backbone_ckpt = args.backbone_ckpt,
            device = args.device
        )
    print(args.backbone_name)

    backbone_res = backbone_pipeline(
        preprocess = None,
        normalizer = None,
        backbone_model = None,
    )

    if "clip_classifier" in args.backbone_name:
        raise NotImplementedError
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = args.backbone_name.split(":")[1]
        backbone = CLIPWrapper(clip_backbone_name)
        preprocess = backbone.preprocess
        backbone.load_state_dict(torch.load(args.backbone_ckpt)["state"])
        backbone = backbone.float()\
                    .to(args.device)\
                    .eval()
    elif "open_clip" in args.backbone_name:
        import open_clip
        clip_backbone_name = args.backbone_name.split(":")[1]
        if os.path.isdir(args.backbone_ckpt):
            backbone, _, preprocess = open_clip.create_model_and_transforms(clip_backbone_name, pretrained='openai', cache_dir=args.backbone_ckpt)
        else:
            backbone, _, preprocess = open_clip.create_model_and_transforms(clip_backbone_name, pretrained=args.backbone_ckpt, cache_dir=model_zoo.CLIP)

        backbone = backbone.float()\
            .to(args.device)\
            .eval()

        backbone_res.backbone_model = backbone
        backbone_res.normalizer = transforms.Compose(preprocess.transforms[-1:])
        backbone_res.preprocess = transforms.Compose(preprocess.transforms[:-1])
        
    elif "clip" in args.backbone_name:
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = args.backbone_name.split(":")[1]
        if os.path.isdir(args.backbone_ckpt):
            backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.backbone_ckpt)
        else:
            backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=model_zoo.CLIP)
            backbone.load_state_dict(torch.load(args.backbone_ckpt)["state_dict"])

        backbone = backbone.float()\
            .to(args.device)\
            .eval()
        
        backbone_res.backbone_model = backbone
        backbone_res.normalizer = transforms.Compose(preprocess.transforms[-1:])
        backbone_res.preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    elif args.backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        if os.path.isdir(args.backbone_ckpt):
            model = ptcv_get_model(args.backbone_name, pretrained=True, root=args.backbone_ckpt)
        else:
            model = ptcv_get_model(args.backbone_name, pretrained=False)
            model.load_state_dict(torch.load(args.backbone_ckpt)["state_dict"])
            
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        backbone.encode_image = lambda image: backbone(image)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
        backbone = backbone.to(args.device)\
            .eval()
        
        backbone_res.backbone_model = backbone
        backbone_res.normalizer = transforms.Compose(preprocess.transforms[-1:])
        backbone_res.preprocess = transforms.Compose(preprocess.transforms[:-1])
        backbone_res.additional_components["model_top"] = model_top
    
    elif args.backbone_name.lower() == "ham10000_inception":
        raise NotImplementedError
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
    else:
        raise NotImplementedError

 
    return backbone_res

@dataclass
class pcbm_configure:
    pcbm_ckpt:str
    device:Union[str, torch.device]

def load_pcbm(args:Union[argparse.Namespace, pcbm_configure], 
              dataset:dataset_collection,
              concepts_bank:ConceptBank) -> PosthocLinearCBM:
    
    if isinstance(args, argparse.Namespace):
        args = pcbm_configure(
            pcbm_ckpt = args.pcbm_ckpt,
            device = args.device
        )
    
    posthoc_layer = PosthocLinearCBM(concepts_bank, 
                                     idx_to_class=dataset.idx_to_class, 
                                     n_classes=dataset.idx_to_class.__len__())
    posthoc_layer.load_state_dict(torch.load(args.pcbm_ckpt, map_location=args.device))
    posthoc_layer.to(args.device)
    return posthoc_layer

@dataclass
class model_pipeline_configure:
    pcbm_arch:str
    pcbm_ckpt:str
    device:Union[str, torch.device]

def build_pcbm_model(args:Union[argparse.Namespace, model_pipeline_configure], 
                         model_context:model_pipeline):
    model = None
    if isinstance(args, argparse.Namespace):
        args = model_pipeline_configure(
            pcbm_arch = args.pcbm_arch,
            pcbm_ckpt = args.pcbm_ckpt,
            device = args.device
        )

    if args.pcbm_arch == "pcbm":
        model = PCBM_Net(model_context=model_context)
        if args.pcbm_ckpt is not None and os.path.exists(args.pcbm_ckpt):
            model.load_state_dict(torch.load(args.pcbm_ckpt), strict=False)
            print(f"Successfully loaded checkpoint from {args.pcbm_ckpt}")
        model.to(args.device)
    elif args.pcbm_arch == "css_pcbm":
        model = css_cbm(model_context.normalizer, 
                        model_context.concept_bank, 
                        model_context.backbone)
        if args.pcbm_ckpt is not None and os.path.exists(args.pcbm_ckpt):
            model.load_state_dict(torch.load(args.pcbm_ckpt), strict=False)
            print(f"Successfully loaded checkpoint from {args.pcbm_ckpt}")
        model.to(args.device)
    else:
        raise NotImplementedError
    
    return model 

def model_forward_wrapper(model_context:model_pipeline,):
    def model_forward(batch_X:torch.Tensor,
                      model_context:model_pipeline,):
        batch_X_normalized = model_context.normalizer(batch_X)
        embeddings = model_context.backbone.encode_image(batch_X_normalized)
        concept_projs = model_context.posthoc_layer.compute_dist(embeddings)
        
        return concept_projs
    
    return partial(model_forward, model_context = model_context)

def load_model_pipeline(args:argparse.Namespace):
    concept_bank = load_concept_bank(args)
    backbone = load_backbone(args)
    dataset = load_dataset(args, backbone.preprocess)
    
    model_context = model_pipeline(concept_bank = concept_bank, 
                   preprocess = backbone.preprocess, 
                   normalizer = backbone.normalizer, 
                   backbone = backbone.backbone_model)
    
    pcbm_model = build_pcbm_model(args, model_context=model_context)

    return concept_bank, backbone, dataset, model_context, pcbm_model

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def topK_concept_to_name(args, pcbm_net:CBM_Net,
                         batch_X:torch.Tensor,
                         K = 5):
    # from model_utils import PCBM_Net
    projs = pcbm_net.encode_as_concepts(batch_X)
    topk_values, topk_indices = torch.topk(projs, K, dim=1)
    predicted_Y = pcbm_net.forward_projs(projs).argmax(1)

    # topk_concept = [{pcbm_net.CAV_layer.names[idx]:round(float(val), 2) for idx, val in zip(irow, vrow)} for irow, vrow in zip(topk_indices, topk_values)]
    # classification_res = [f"{pcbm_net.idx_to_class[Y.item()]}" for Y in predicted_Y]
    # print(f"top (K = {K}) concepts: {json.dumps(topk_concept, indent=4)}")
    # print(f"classification result: {json.dumps(classification_res, indent=4)}")


def get_topK_concept_logit(args, batch_X:torch.Tensor,  
                            batch_Y:torch.Tensor,
                            model_context:model_pipeline,
                            K:int = 5,):
    
    batch_X_normalized = model_context.normalizer(batch_X)
    embeddings = model_context.backbone.encode_image(batch_X_normalized)
    projs = model_context.posthoc_layer.compute_dist(embeddings)
    predicted_Y = model_context.posthoc_layer.forward_projs(projs)
    accuracy = (predicted_Y.argmax(1) == batch_Y).float().mean().item()
    
    topk_values, topk_indices = torch.topk(projs, K, dim=1)
    topk_concept = [{model_context.posthoc_layer.names[idx]:round(float(val), 2) for idx, val in zip(irow, vrow)} for irow, vrow in zip(topk_indices, topk_values)]
    classification_res = [f"{model_context.posthoc_layer.idx_to_class[Y.item()]} -> {model_context.posthoc_layer.idx_to_class[Y_hat.item()]}" for Y, Y_hat in zip(batch_Y, predicted_Y.argmax(1))]
    print(f"top (K = {K}) concepts: {json.dumps(topk_concept, indent=4)}")
    print(f"classification result: {json.dumps(classification_res, indent=4)}")
    print(f"accuracy: {accuracy}")


def evaluzate_accuracy(args, batch_X:torch.Tensor, 
                            batch_Y:torch.Tensor,
                            model_context:model_pipeline,):
    batch_X_normalized = model_context.normalizer(batch_X)
    embeddings = model_context.backbone.encode_image(batch_X_normalized)
    projs = model_context.posthoc_layer.compute_dist(embeddings)
    predicted_Y = model_context.posthoc_layer.forward_projs(projs)
    accuracy = (predicted_Y.argmax(1) == batch_Y).float().mean().item()
    
    return accuracy