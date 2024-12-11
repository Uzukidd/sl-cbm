import argparse
import time

import heapq
import clip
import pickle as pkl


import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *

from utils import *

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    parser.add_argument("--concept-bank", default="/home/ksas/Public/datasets/rival10_concept_bank/multimodal_concept_clip:RN50_rival10_recurse:1.pkl", type=str, help="Path to the concept bank")
    parser.add_argument("--pcbm-ckpt", default="data/ckpt/RIVAL_10/pcbm_RIVAL10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt", type=str, help="Path to the PCBM checkpoint")

    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", required=True, type=str)

    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--concept-pooling", default="max_pooling_class_wise", type=str)
    
    parser.add_argument("--dataset", default="rival10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument('--save-100-local', action='store_true')

    return parser.parse_args()

def select_act_topK_image(args, model:nn.Module, data_loader:DataLoader, num_classes:int=10, K:int=10):
    classwise_topK_image = [[(-np.inf, None) for j in range(K)] for i in range(num_classes)]
    classwise_attr_label = [None for i in range(K)]
    classwise_attr_mask = [None for i in range(K)]

    for idx, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            image:torch.Tensor =  data["img"].to(args.device)
            attr_labels:torch.Tensor = data["attr_labels"].to(args.device)
            attr_masks:torch.Tensor = data["attr_masks"][:, :-1, 0:1, :, :].to(args.device)
            class_label:torch.Tensor = data["og_class_label"].to(args.device)

            logit:torch.Tensor = model(image)
            logit = logit[torch.arange(0, logit.size(0)).long().to(args.device), class_label]

            for ind_mask in range(image.size(0)):
                heapq.heappushpop(classwise_topK_image[class_label[ind_mask].item()],
                                  (logit[ind_mask].item(), idx, 
                                   (image[ind_mask].detach().cpu(), 
                                    attr_labels[ind_mask].detach().cpu(), 
                                    attr_masks[ind_mask].detach().cpu())))

    classwise_attr_label = [[x[2][1] for x in classwise_topK_image[i]] for i in range(num_classes)]
    classwise_attr_mask = [[x[2][2] for x in classwise_topK_image[i]] for i in range(num_classes)]
    classwise_topK_image = [[x[2][0] for x in classwise_topK_image[i]] for i in range(num_classes)]
    
    return classwise_topK_image, classwise_attr_label, classwise_attr_mask



class get_concept_net_func:
    
    @staticmethod
    def clip_classifier(args, model_context:model_pipeline):
        return model_context.backbone
    
    @staticmethod
    def open_clip(args, model_context:model_pipeline):
        posthoc_concept_net = PCBM_Net(model_context=model_context)
        posthoc_concept_net.output_type("concepts")
        return posthoc_concept_net

    @staticmethod
    def clip(args, model_context:model_pipeline):
        posthoc_concept_net = PCBM_Net(model_context=model_context)
        posthoc_concept_net.output_type("concepts")
        return posthoc_concept_net

class select_concept_func:
    
    @staticmethod
    def clip_classifier(args, concept_bank:ConceptBank,
                        model_context:model_pipeline,
                        dataset:dataset_collection,
                        explain_algorithm:GradientAttribution,
                        explain_algorithm_forward:Callable,):
        
        try:
            with open('.cache/classwise_topK_image, classwise_attr_label, classwise_attr_mask.pkl', 'rb') as f:
                classwise_topK_image, classwise_attr_label, classwise_attr_mask = pickle.load(f)
        except:
            classwise_topK_image, classwise_attr_label, classwise_attr_mask = select_act_topK_image(args, model_context.backbone,  
                                                                                                    dataset.test_loader)
            with open('.cache/classwise_topK_image, classwise_attr_label, classwise_attr_mask.pkl', 'wb') as f:
                pkl.dump((classwise_topK_image, classwise_attr_label, classwise_attr_mask), f)
        concept_net = model_context.backbone
        concepts = concept_bank.concept_info.concept_names
        
        # Fixed size
        embedding_size = 512
        iou_matrix = torch.zeros((embedding_size), concepts.__len__())
        for ind_class_img, ind_class_attr, ind_class_attr_masks in tqdm(zip(classwise_topK_image, classwise_attr_label, classwise_attr_mask)):
            for ind_X, ind_attr, ind_attr_masks in zip(ind_class_img, ind_class_attr, ind_class_attr_masks):
                iou_matrix += interpret_ind_X(args, ind_X, ind_attr, 
                                ind_attr_masks, 
                                explain_algorithm, 
                                explain_algorithm_forward, 
                                torch.arange(0, embedding_size).to(args.device))

        return iou_matrix.argmax(0).to(args.device)

    @staticmethod
    def clip(args, concept_bank:ConceptBank,
            model_context:model_pipeline,
            dataset:dataset_collection,
            explain_algorithm:GradientAttribution,
            explain_algorithm_forward:Callable,):
        K = concept_bank.concept_info.concept_names.__len__()
        return torch.arange(0, K).to(args.device)
    
    @staticmethod
    def open_clip(args, concept_bank:ConceptBank,
            model_context:model_pipeline,
            dataset:dataset_collection,
            explain_algorithm:GradientAttribution,
            explain_algorithm_forward:Callable,):
        K = concept_bank.concept_info.concept_names.__len__()
        return torch.arange(0, K).to(args.device)

def show_mask(mask):
    mask =  mask.permute(1, 2, 0).detach().cpu().numpy()

    plt.imshow(mask)
    plt.axis('off')
    plt.show()
    
def interpret_ind_X(args, ind_X, ind_attr, ind_attr_masks, explain_algorithm:GradientAttribution,
                          explain_algorithm_forward:Callable,
                          explain_concept:torch.Tensor):
    ind_X = ind_X.to(args.device).unsqueeze(0)
    ind_attr_masks = ind_attr_masks.to(args.device)
    iou_matrix = torch.empty((explain_concept.size(0), ind_attr_masks.size(0)))
    for splited_mask in range(explain_concept.size(0)):
        attribution = explain_algorithm_forward(
            batch_X = ind_X,
            explain_algorithm = explain_algorithm,
            target = explain_concept[splited_mask:splited_mask + 1]
        )

        iou, dice = attribution_iou(attribution.expand(ind_attr_masks.size(0), -1, -1, -1), ind_attr_masks)
        iou_matrix[splited_mask] = iou.cpu()

    return iou_matrix
    

def interpret_all_concept(args, 
                        #   batch_X:torch.Tensor,
                        #   concept_net:Callable[..., torch.Tensor],
                          
                          data_loader:DataLoader,
                          explain_algorithm:GradientAttribution,
                          explain_algorithm_forward:Callable,
                          explain_concept:torch.Tensor):
    
    attrwise_iou = None
    attrwise_amount = None
    for idx, data in enumerate(tqdm(data_loader)):
        image:torch.Tensor =  data["img"].to(args.device)
        attr_labels:torch.Tensor = data["attr_labels"].to(args.device)
        attr_masks:torch.Tensor = data["attr_masks"][:, :-1, 0:1, :, :].to(args.device)
        class_label:torch.Tensor = data["og_class_label"].to(args.device)
        class_name:torch.Tensor = data["og_class_name"]

        B, C, W, H = image.size()
        _, K = attr_labels.size()

        for batch_mask in range(B):
            ind_X = image[batch_mask:batch_mask+1]
            ind_attr_labels = attr_labels[batch_mask]
            ind_attr_masks = attr_masks[batch_mask]
            ind_class_name = class_name[batch_mask]

            attribution = explain_algorithm_forward(
                batch_X = ind_X,
                explain_algorithm = explain_algorithm,
                target = explain_concept
            )

            iou, dice = attribution_iou(attribution, ind_attr_masks)

            if attrwise_iou is None:
                attrwise_iou = image.new_zeros(K)
                attrwise_amount = image.new_zeros(K)
            
            attrwise_amount += ind_attr_labels
            attrwise_iou += iou * ind_attr_labels
    
    attrwise_iou = attrwise_iou / attrwise_amount
    return attrwise_iou

            
def main(args):
    set_random_seed(args.universal_seed)
    concept_bank = load_concept_bank(args)
    backbone, preprocess = load_backbone(args)
    normalizer = transforms.Compose(preprocess.transforms[-1:])
    preprocess = transforms.Compose(preprocess.transforms[:-1])
    
    dataset = load_dataset(args, preprocess)
    dataset.trainset.masks_dict = True
    dataset.testset.masks_dict = True

    posthoc_layer = load_pcbm(args, dataset, concept_bank)

    model_context = model_pipeline(concept_bank = concept_bank, 
                   posthoc_layer = posthoc_layer, 
                   preprocess = preprocess, 
                   normalizer = normalizer, 
                   backbone = backbone)

    concept_net = getattr(get_concept_net_func, args.backbone_name.split(":")[0])(args, model_context)

    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory, args.explain_method)(posthoc_concept_net = concept_net)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    attribution_pooling:Callable[..., torch.Tensor] = getattr(attribution_pooling_forward, args.concept_pooling)
    
    explain_concept:torch.Tensor = getattr(select_concept_func, args.backbone_name.split(":")[0])(args, concept_bank,
                                                                                                  model_context,
                                                                                                  dataset,
                                                                                                  explain_algorithm,
                                                                                                  explain_algorithm_forward,)
    
    attrwise_iou = interpret_all_concept(args, dataset.test_loader, 
                          explain_algorithm, 
                          explain_algorithm_forward,
                          explain_concept)
    for names, iou in zip(concept_bank.concept_info.concept_names, attrwise_iou):
        print(f"{names}: {iou:.2f}")
    # classwise_topK_image = select_act_topK_image(args, backbone, dataset.test_loader)

    # images = np.array([[img.permute(1, 2, 0).numpy() for img in row] for row in classwise_topK_image])

    # fig, axes = plt.subplots(10, 10, figsize=(10, 10))

    # for i in range(10):
    #     for j in range(10):
    #         axes[i, j].imshow(images[i, j], cmap='gray')
    #         axes[i, j].axis('off')

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()

    # accuracy = []
    # for data in tqdm(dataset.test_loader):
    #     image, label =  data["img"].to(args.device), data["og_class_label"].to(args.device)
    #     accuracy.append((backbone(image).argmax(1) == label).float().mean().item())
    # print(np.array(accuracy).mean())
if __name__ == "__main__":
    args = config()
    print(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        print(f"GPU devices failed. Change to {args.device}")
    main(args)
    