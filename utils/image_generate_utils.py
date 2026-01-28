import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from itertools import islice

from .model_utils import *
from .visual_utils import *
from .constants import *
from .explain_utils import *
from .common_utils import *
from .intervene_utils import *
from .eval_utils import *

def attribution2img(batch_X:torch.Tensor, attributions:list[torch.Tensor], blur=True):
    batch_X = reduce_tensor_as_numpy(batch_X)
    attributions = [reduce_tensor_as_numpy(attribution) for attribution in attributions]
    
    attn_map = []
    for attribution in attributions:
        attn_map.append(getAttMap(batch_X, attribution.sum(2), blur))

    res = [np.clip(batch_X, 0.0, 1.0)]
    for idx, map in enumerate(attn_map):
        res.append(np.clip(map, 0.0, 1.0))
    
    return res

def image_generate(
    args,
    model: CBM_Net,
    preprocess: transforms.Compose,
    dataset: dataset_collection,
    concept_bank: ConceptBank,
    explain_method: str,
    image_count: int):
    
    explain_algorithm: GradientAttribution = getattr(
        model_explain_algorithm_factory, explain_method
    )(forward_func=model.encode_as_concepts, model=model)
    
    explain_algorithm_forward = partial(
        getattr(
        model_explain_algorithm_forward, explain_method
        ), explain_algorithm=explain_algorithm
    )

    image_save_to = os.path.join(args.save_path, "images.pt")
    # if not os.path.exists(image_save_to):
    #     os.mkdir(image_save_to)
        
        
    topk_concept_indice, topk_concept_weights = (
        model.get_topK_concepts()
    )  # [C, K], [C, K]
            
    explain_concept: torch.Tensor = torch.arange(
        0, concept_bank.concept_info.concept_names.__len__()
    ).to(args.device)

    data_loader = dataset.train_loader
    res = []
    
    for idx, data in enumerate(tqdm(islice(data_loader, image_count))):
        image: torch.Tensor = data["img"].to(args.device)
        attr_labels: torch.Tensor = data["attr_labels"].to(args.device)
        attr_masks: torch.Tensor = (
            data["attr_masks"][:, :-1, :, :, :]
            .amax(dim=2, keepdim=True)
            .to(args.device)
        )
        class_label: torch.Tensor = data["og_class_label"].to(args.device)
        class_name: torch.Tensor = data["og_class_name"]

        B, C, W, H = image.size()
        _, K = attr_labels.size()
        with torch.no_grad():
            class_prediction, concept_predictions, _ = model(image)

        for batch_mask in range(B):
            ind_X = image[batch_mask : batch_mask + 1]
            ind_attr_labels = attr_labels[batch_mask]
            ind_attr_masks = attr_masks[batch_mask]
            ind_class_name = class_name[batch_mask]
            ind_class_label = class_label[batch_mask].item()
            ind_class_prediction = class_prediction[batch_mask]

            # Get active labels
            k_val = int(torch.sum(ind_attr_labels).item())
            valid_concepts_mask = torch.zeros_like(ind_attr_labels)
            _, top_pred_indices = torch.topk(
                concept_predictions[batch_mask], k=k_val, dim=-1
            )
            valid_concepts_mask[top_pred_indices] = 1
            valid_concepts_mask = valid_concepts_mask & ind_attr_labels

            # Explain concepts
            model.zero_grad()
            concepts_attribution: torch.Tensor = explain_algorithm_forward(
                batch_X=ind_X, target=explain_concept
            )  # [18, 1, H, W]

            classes_attribution: torch.Tensor = CBM_Net.attribute_weighted_class(
                concepts_attribution,
                topk_concept_weights[ind_class_label],
                topk_concept_indice[ind_class_label],
            )
            # [1, 1, H, W]
            batch_attribution = torch.concat([concepts_attribution, classes_attribution], dim = 0)
            binarized_batch_attribution = torch.maximum(
                batch_attribution, batch_attribution.new_zeros(batch_attribution.size())
            )
            batch_attribution = normalize(batch_attribution)
            
            res.append({
                "img": attribution2img(ind_X, binarized_batch_attribution),
                "ind_class_name": ind_class_name,
                "ind_class_label": ind_class_label,
                "ind_class_prediction": ind_class_prediction.argmax(-1)
            })
    
    torch.save(res, image_save_to)
    return res