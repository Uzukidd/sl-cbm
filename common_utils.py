import argparse
import random
import numpy as np
import pickle as pkl
import json
from tqdm import tqdm
from typing import Tuple, Callable, Union, Optional
from dataclasses import dataclass
from functools import partial


import clip
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, PosthocHybridCBM, get_model
from pcbm.training_tools import load_or_compute_projections

from constants import dataset_cosntants


@dataclass
class model_result:
    batch_X:Optional[torch.Tensor] = None
    batch_Y:Optional[torch.Tensor] = None
    embeddings:Optional[torch.Tensor] = None
    concept_projs:Optional[torch.Tensor] = None
    batch_Y_predicted:Optional[torch.Tensor] = None


@dataclass
class model_pipeline:
    concept_bank:ConceptBank
    posthoc_layer:PosthocLinearCBM
    preprocess:transforms.Compose
    normalizer:transforms.Compose
    backbone:nn.Module


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
    import torch
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

def load_dataset(args, preprocess:transforms.Compose):
    trainset, testset = None, None
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=dataset_cosntants.CIFAR10_DIR, train=True,
                                    download=False, transform=preprocess)
        testset = datasets.CIFAR10(root=dataset_cosntants.CIFAR10_DIR, train=False,
                                    download=False, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
    
    
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=dataset_cosntants.CIFAR100_DIR, train=True,
                                    download=False, transform=preprocess)
        testset = datasets.CIFAR100(root=dataset_cosntants.CIFAR100_DIR, train=False,
                                    download=False, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)


    elif args.dataset == "cub":
        from pcbm.data.cub import load_cub_data
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(dataset_cosntants.CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(dataset_cosntants.CUB_PROCESSED_DIR, "test.pkl")
        train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=dataset_cosntants.CUB_DATA_DIR, resol=224, normalizer=None,
            n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=dataset_cosntants.CUB_DATA_DIR, resol=224, normalizer=None,
                n_classes=num_classes, resampling=True)

        classes = open(os.path.join(dataset_cosntants.CUB_DATA_DIR, "classes.txt")).readlines()
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


    else:
        raise ValueError(args.dataset)

    return trainset, testset, class_to_idx, idx_to_class, train_loader, test_loader

def load_concept_bank(args) -> ConceptBank:
    all_concepts = pkl.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)
    
    return concept_bank

class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def load_backbone(args) -> Tuple[nn.Module, transforms.Compose]:
    if "clip" in args.backbone_name:
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = args.backbone_name.split(":")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.backbone_ckpt)
        backbone = backbone.float()\
                    .to(args.device)\
                    .eval()
    
    elif args.backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(args.backbone_name, pretrained=True, root=args.backbone_ckpt)
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

 
    return backbone, preprocess

def load_pcbm(args) -> PosthocLinearCBM:
    posthoc_layer:PosthocLinearCBM = torch.load(args.pcbm_ckpt, map_location=args.device)
    # print(posthoc_layer.analyze_classifier(k=5))
    # print(posthoc_layer.names)
    # print(posthoc_layer.names.__len__())
    return posthoc_layer

def model_forward_wrapper(model_context:model_pipeline,):
    def model_forward(batch_X:torch.Tensor,
                      model_context:model_pipeline,):
        batch_X_normalized = model_context.normalizer(batch_X)
        embeddings = model_context.backbone.encode_image(batch_X_normalized)
        concept_projs = model_context.posthoc_layer.compute_dist(embeddings)
        
        return concept_projs
    
    return partial(model_forward, model_context = model_context)

def get_topK_concept_logit(args, batch_X:torch.Tensor, 
                            batch_Y:torch.Tensor,
                            model_context:model_pipeline,
                            K:int = 5,):
    
    batch_X_normalized = model_context.normalizer(batch_X)
    embeddings = model_context.backbone.encode_image(batch_X_normalized)
    projs = model_context.posthoc_layer.compute_dist(embeddings)
    predicted_Y = model_context.posthoc_layer.forward_projs(projs)
    accuracy = (predicted_Y.argmax(1) == batch_Y).float().mean().item()
    
    topk_values, topk_indices = torch.topk(projs, 5, dim=1)
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