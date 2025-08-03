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
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument('--dataset-scalar', default=None, type=float)
    parser.add_argument('--intervention', action='store_true')


    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)
    parser.add_argument('--batch-vis', action='store_true')

    return parser.parse_args()


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

# viz_attn(batch_X,
#         attributions,
#         blur=True,
#         save_to=None)

def __vis_ind_image(ind_X:torch.Tensor, 
                    batch_attribution:torch.Tensor,
                    batch_attr_mask:torch.Tensor,
                    concept:int,
                    prefix:str,
                    save_to:str):
    eps=1e-10
    def binarize(m):
        m = m.clone()
        m[torch.isnan(m)] = 0

        max_val = torch.amax(m, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        max_val[max_val < eps] = eps
        m = m / max_val
        m[m>0.5] = 1
        m[m<0.5] = 0
        return m

    
    binarized_batch_attribution = torch.maximum(batch_attribution, 
                                                batch_attribution.new_zeros(batch_attribution.size()))
    binarized_batch_attribution = binarize(binarized_batch_attribution)

    binarized_batch_attr_mask = batch_attr_mask

    viz_attn_multiple(ind_X[0],
        [binarized_batch_attr_mask[concept], binarized_batch_attribution[concept]],
        prefix = prefix,
        save_to = save_to)

            
def main(args):
    set_random_seed(args.universal_seed)
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    model.eval()
    
    # val_acc, val_concept_acc = val_one_epoch(dataset.test_loader, model, args.device)
    # args.logger.info("\t Val Class Accuracy = {} and Val Concept Accuracy = {}.".format(round(val_acc,2),round(val_concept_acc,2)))

    # Start Rival attrbution alignment evaluation
    eval_model_explainability(
        args,
        model,
        backbone.preprocess,
        dataset,
        concept_bank,
        args.explain_method,
    )

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
