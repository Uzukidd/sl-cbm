import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
from tqdm import tqdm

from rival10.constants import RIVAL10_constants
from .model_utils import *
from .visual_utils import *
from .constants import *
from .explain_utils import *
from .common_utils import *
from typing import Callable, Dict, Tuple

eps = 1e-10


def binarize(m: torch.Tensor):
    m = m.clone()
    m[m > 0.5] = 1
    m[m < 0.5] = 0
    return m


def normalize(m: torch.Tensor):
    m = m.clone()
    m[torch.isnan(m)] = 0

    max_val = torch.amax(m, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    max_val[max_val < eps] = eps
    m = m / max_val

    return m


class attribution_metric:
    __ALL_METRIC__ = ["iou", "dice", "prec_iou"]

    def __init__(self, name: str, num_labels: int, num_concepts: int):
        self.name: str = name
        self.amount: Dict[str, torch.Tensor] = {
            k: torch.zeros((num_labels, num_concepts)) for k in self.__ALL_METRIC__
        }  # 3 x [N, C]
        self.value: Dict[str, torch.Tensor] = {
            k: torch.zeros((num_labels, num_concepts)) for k in self.__ALL_METRIC__
        }

    def load_from_dict(self, formatted_dict: Dict):
        self.name: str = formatted_dict["name"]
        self.amount: Dict[str, torch.Tensor] = formatted_dict["amount"]
        self.value: Dict[str, torch.Tensor] = formatted_dict["value"]

    @staticmethod
    def pack_tuple(attribution: Tuple[torch.Tensor]):
        res_dict = {
            name: metric
            for metric, name in zip(attribution, attribution_metric.__ALL_METRIC__)
        }
        return res_dict

    def update(
        self,
        label: int,
        attribution_metric: Dict[str, torch.Tensor],
        attribution_mask: torch.Tensor,
        valid_attribution_mask: torch.Tensor,
    ):
        for metric in self.__ALL_METRIC__:
            # Align tensor device
            if self.amount[metric].device != attribution_metric[metric].device:
                self.amount[metric] = self.amount[metric].to(
                    attribution_metric[metric].device
                )
                self.value[metric] = self.value[metric].to(
                    attribution_metric[metric].device
                )

            self.amount[metric][label] += attribution_mask
            self.value[metric][label] += (
                attribution_metric[metric] * valid_attribution_mask
            )

    def load_from_path(self, path: str):
        format_dict = torch.load(path)
        self.name = format_dict["name"]

        # self.amount = {k:v for k, v in format_dict["amount"].items()}
        # self.value = {k:v for k, v in format_dict["value"].items()}

        self.amount = {k: torch.from_numpy(v) for k, v in format_dict["amount"].items()}
        self.value = {k: torch.from_numpy(v) for k, v in format_dict["value"].items()}

    def format_dict(self):
        return {
            "name": self.name,
            "amount": {k: v.detach().cpu().numpy() for k, v in self.amount.items()},
            "value": {k: v.detach().cpu().numpy() for k, v in self.value.items()},
        }

    def format_output(
        self,
        class_name: list[str],
        concept_name: list[str],
        full_text: bool = False,
        latex: bool = False,
    ) -> str:
        totall_metric = {}
        all_metric = {}

        for metric in self.__ALL_METRIC__:
            total_amount = self.amount[metric].sum(dim=0)
            total_value = self.value[metric].sum(dim=0)
            totall_metric[metric] = total_value / total_amount

            all_metric[metric] = self.value[metric].sum() / self.amount[metric].sum()

        formatted_str = None

        if latex:
            metric_template = "{name:16}\t{iou:.2f} &\t{dice:.2f} &\t{prec_iou:.2f}\n"
            formatted_str = metric_template.format(
                name=self.name, **{k: t for k, t in all_metric.items()}
            )
        else:
            header = "\t\t\tIoU\tDice\tPrecIoU\n"
            metric_template = "{name:16}\t{iou:.2f}\t{dice:.2f}\t{prec_iou:.2f}\n"

            formatted_str = f"{self.name}\n" + header + ""

            for ind, concepts_name in enumerate(concept_name):
                formatted_str += metric_template.format(
                    name=concepts_name, **{k: t[ind] for k, t in totall_metric.items()}
                )

            formatted_str += metric_template.format(
                name="Totall", **{k: t for k, t in all_metric.items()}
            )

        return formatted_str


class class_attribution_metric(attribution_metric):
    def __init__(self, name: str, num_labels: int):
        super().__init__(name, num_labels, 1)

    @staticmethod
    def pack_tuple(attribution: Tuple[torch.Tensor]):
        res_dict = {
            name: metric
            for metric, name in zip(attribution, attribution_metric.__ALL_METRIC__)
        }
        return res_dict

    def update(self, label: int, attribution_metric: Dict[str, torch.Tensor]):
        for metric in self.__ALL_METRIC__:
            # Align tensor device
            if self.amount[metric].device != attribution_metric[metric].device:
                self.amount[metric] = self.amount[metric].to(
                    attribution_metric[metric].device
                )
                self.value[metric] = self.value[metric].to(
                    attribution_metric[metric].device
                )

            self.amount[metric][label] += 1
            self.value[metric][label] += attribution_metric[metric]

    def load_from_path(self, path: str):
        format_dict = torch.load(path)
        self.name = format_dict["name"]

        # self.amount = {k:v for k, v in format_dict["amount"].items()}
        # self.value = {k:v for k, v in format_dict["value"].items()}

        self.amount = {k: torch.from_numpy(v) for k, v in format_dict["amount"].items()}
        self.value = {k: torch.from_numpy(v) for k, v in format_dict["value"].items()}

    def format_dict(self):
        return {
            "name": self.name,
            "amount": {k: v.detach().cpu().numpy() for k, v in self.amount.items()},
            "value": {k: v.detach().cpu().numpy() for k, v in self.value.items()},
        }

    def format_output(
        self, classes_name: list[str], full_text: bool = False, latex: bool = False
    ) -> str:
        totall_metric = {}
        all_metric = {}

        for metric in self.__ALL_METRIC__:
            total_amount = self.amount[metric].sum(dim=1)
            total_value = self.value[metric].sum(dim=1)
            totall_metric[metric] = total_value / total_amount

            all_metric[metric] = self.value[metric].sum() / self.amount[metric].sum()

        formatted_str = None

        if latex:
            metric_template = "{name:16}\t{iou:.2f} &\t{dice:.2f} &\t{prec_iou:.2f}\n"
            formatted_str = metric_template.format(
                name=self.name, **{k: t for k, t in all_metric.items()}
            )
        else:
            header = "\t\t\tIoU\tDice\tPrecIoU\n"
            metric_template = "{name:16}\t{iou:.2f}\t{dice:.2f}\t{prec_iou:.2f}\n"

            formatted_str = f"{self.name}\n" + header + ""

            for ind, class_name in enumerate(classes_name):
                formatted_str += metric_template.format(
                    name=class_name, **{k: t[ind] for k, t in totall_metric.items()}
                )

            formatted_str += metric_template.format(
                name="Totall", **{k: t for k, t in all_metric.items()}
            )

        return formatted_str


def attribution_iou(
    batch_attribution: torch.Tensor,
    batch_attr_mask: torch.Tensor,
    eps=1e-10,
    vis: bool = False,
    ind_X: torch.Tensor = None,
    only_positive: bool = True,
    binarize_or_not: bool = False,
):
    """
    args:
        batch_attribution: [B, ...] (non-binarized/non-positive)
        batch_attr_mask: [B, ...] (non-binarized/non-positive)
    """
    if only_positive:
        batch_attribution = torch.maximum(
            batch_attribution, batch_attribution.new_zeros(batch_attribution.size())
        )

    batch_attribution = normalize(batch_attribution)

    if binarize_or_not:
        batch_attribution = binarize(batch_attribution)

    intersection = batch_attribution * batch_attr_mask
    intersection = torch.sum(intersection, dim=(1, 2, 3))

    union = torch.sum(torch.max(batch_attribution, batch_attribution), dim=(1, 2, 3))
    attribution_area = torch.sum(batch_attribution, dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + intersection + eps)
    iou = (intersection + eps) / (union + eps)
    prec_iou = (intersection + eps) / (attribution_area + eps)

    return iou.detach(), dice.detach(), prec_iou.detach()


def __vis_ind_image(
    ind_X: torch.Tensor,
    batch_attribution: torch.Tensor,
    batch_attr_mask: torch.Tensor,
    concept: int,
    prefix: str,
    save_to: str,
):
    binarized_batch_attribution = torch.maximum(
        batch_attribution, batch_attribution.new_zeros(batch_attribution.size())
    )
    batch_attribution = normalize(batch_attribution)

    # binarized_batch_attribution = binarize(binarized_batch_attribution)

    binarized_batch_attr_mask = batch_attr_mask

    viz_attn_multiple(
        ind_X[0],
        [binarized_batch_attr_mask[concept], binarized_batch_attribution[concept]],
        prefix=prefix,
        save_to=save_to,
    )


def collect_class_attribution(
    batch_attribution: torch.Tensor,
    concepts_indices: torch.Tensor,
    concepts_weights: torch.Tensor,
    label: int,
):
    """
    args:
        batch_attribution: [B, ...] (non-binarized/non-positive)
        batch_label: [B, 1]
    """
    binarized_batch_attribution = torch.maximum(
        batch_attribution, batch_attribution.new_zeros(batch_attribution.size())
    )
    binarized_batch_attribution = binarize(binarized_batch_attribution)

    class_indices = concepts_indices[label]  # [K]
    class_weights = concepts_weights[label]  # [K]

    selected_attribution = batch_attribution[class_indices]
    weighted_attribution = torch.sum(
        selected_attribution * class_weights, dim=0, keepdim=True
    )

    return weighted_attribution


def save_key_image(
    ind_class_name: str,
    ind_X: torch.Tensor,
    attribution: torch.Tensor,
    ind_attr_masks: torch.Tensor,
    prefix: str,
    save_to: str,
):
    if ind_class_name in RIVAL10_constants._KEY_FEATURES:
        for attr_name in RIVAL10_constants._KEY_FEATURES[ind_class_name]:
            attr_label = RIVAL10_constants._ALL_ATTRS.index(attr_name)
            __vis_ind_image(
                ind_X,
                attribution,
                ind_attr_masks,
                attr_label,
                f"{ind_class_name}-{attr_name}-{prefix}",
                save_to,
            )


def save_best_image(
    ind_class_name: str,
    ind_X: torch.Tensor,
    attribution: torch.Tensor,
    ind_attr_masks: torch.Tensor,
    best_iou: torch.Tensor,
    iou: torch.Tensor,
    save_to: str,
):
    if ind_class_name in RIVAL10_constants._KEY_FEATURES:
        for attr_name in RIVAL10_constants._KEY_FEATURES[ind_class_name]:
            attr_label = RIVAL10_constants._ALL_ATTRS.index(attr_name)
            if best_iou[attr_label] <= iou[attr_label]:
                __vis_ind_image(
                    ind_X,
                    attribution,
                    ind_attr_masks,
                    attr_label,
                    f"{ind_class_name}-{attr_name}-best",
                    save_to,
                )


def interpret_all_concept(
    args,
    model: CBM_Net,
    data_loader: DataLoader,
    explain_algorithm_forward: Callable,
    explain_concept: torch.Tensor,
):
    concepts_segmentation_metric = attribution_metric(
        "cocnepts_segmentation_metric",
        num_labels=model.get_num_classes(),
        num_concepts=model.get_num_concepts(),
    )  # only-posistive, binarized

    classes_segmentation_metric = class_attribution_metric(
        "classes_segmentation_metric", num_labels=model.get_num_classes()
    )  # only-posistive, binarized

    weighted_cocnepts_metric = attribution_metric(
        "weighted_concepts_metric",
        num_labels=model.get_num_classes(),
        num_concepts=model.get_num_concepts(),
    )  # only-posistive, not-binarized
    # Save saliency map as pth file if needed.
    # saliency_map_pack = None
    # if save_saliency_map:
    #     saliency_map_pack = {
    #         "concepts_saliency_map": [],
    #         "classes_saliency_map": [],
    #     }

    # Assign saving path
    concept_save_to = os.path.join(args.save_path, "images", "concept")
    class_save_to = os.path.join(args.save_path, "images", "class")

    # Prepare weighted class attribution
    topk_concept_indice, topk_concept_weights = (
        model.get_topK_concepts()
    )  # [C, K], [C, K]

    for idx, data in enumerate(tqdm(data_loader)):
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
            _, concept_predictions, _ = model(image)

        for batch_mask in range(B):
            ind_X = image[batch_mask : batch_mask + 1]
            ind_attr_labels = attr_labels[batch_mask]
            ind_attr_masks = attr_masks[batch_mask]
            ind_class_name = class_name[batch_mask]
            ind_class_label = class_label[batch_mask].item()

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

            # # Save saliency map while remaining batch dimension
            # if save_saliency_map:
            #     saliency_map_pack["concepts_saliency_map"].append(concepts_attribution.detach().cpu())
            #     saliency_map_pack["classes_saliency_map"].append(classes_attribution.detach().cpu())

            # ----------
            # Concepts segmentation metric
            # ----------
            concepts_metric = attribution_iou(
                concepts_attribution.sum(dim=1, keepdim=True),
                ind_attr_masks,
                ind_X=ind_X,
                binarize_or_not=True,
            )
            concepts_metric = attribution_metric.pack_tuple(concepts_metric)
            concepts_segmentation_metric.update(
                ind_class_label, concepts_metric, ind_attr_labels, valid_concepts_mask
            )

            # ----------
            # Weighted concepts segmentation metric
            # ----------
            weighted_metric = attribution_iou(
                concepts_attribution.sum(dim=1, keepdim=True),
                ind_attr_masks,
                ind_X=ind_X,
                binarize_or_not=False,
            )
            weighted_metric = attribution_metric.pack_tuple(weighted_metric)
            weighted_cocnepts_metric.update(
                ind_class_label, weighted_metric, ind_attr_labels, valid_concepts_mask
            )

            # ----------
            # Classes segmentation metric
            # ----------
            classes_metric = attribution_iou(
                classes_attribution,
                ind_attr_masks[-1:],
                ind_X=ind_X,
                binarize_or_not=True,
            )
            classes_metric = attribution_metric.pack_tuple(classes_metric)
            classes_segmentation_metric.update(ind_class_label, classes_metric)

            # Save preview image
            if idx < 15:
                save_key_image(
                    ind_class_name,
                    ind_X,
                    concepts_attribution.sum(dim=1, keepdim=True),
                    ind_attr_masks,
                    f"{idx}:{batch_mask}",
                    concept_save_to,
                )
                __vis_ind_image(
                    ind_X,
                    classes_attribution,
                    ind_attr_masks,
                    -1,
                    f"{ind_class_name}-{idx}:{batch_mask}",
                    class_save_to,
                )

    # if save_saliency_map:
    #     saliency_map_pack["concepts_saliency_map"] = torch.stack(saliency_map_pack["concepts_saliency_map"])  #[N, 18, 1, H, W]
    #     saliency_map_pack["classes_saliency_map"] =  torch.stack(saliency_map_pack["classes_saliency_map"])  #[N, 1, 1, H, W]

    return (
        concepts_segmentation_metric,
        classes_segmentation_metric,
        weighted_cocnepts_metric,
    )


def compute_adi(
    args,
    model: CBM_Net,
    data_loader: DataLoader,
    explain_algorithm_forward: Callable,
):
    avg_drop_list = []
    avg_inc_list = []
    avg_gain_list = []
    for idx, data in enumerate(tqdm(data_loader)):
        if isinstance(data, dict):
            image: torch.Tensor = data["img"].to(args.device)
            attr_labels: torch.Tensor = data["attr_labels"].to(args.device)
        else:
            image: torch.Tensor = data[0].to(args.device)
            attr_labels = data[2]

            if isinstance(attr_labels, list):
                attr_labels = torch.stack(attr_labels).permute((1, 0))
                
            attr_labels = attr_labels.to(args.device)
                
        if image.shape.__len__() > 4:
            image = image.view(-1, *image.shape[-3:])
            attr_labels = attr_labels.view(-1, attr_labels.shape[-1])

        B, C, W, H = image.size()
        _, K = attr_labels.size()

        for batch_mask in range(B):
            ind_X = image[batch_mask : batch_mask + 1]  # [1, C, H, W]
            ind_attr_labels = attr_labels[batch_mask]
            if torch.all(ind_attr_labels == 0):
                continue
            # Explain concepts
            model.zero_grad()
            concepts_attribution: torch.Tensor = explain_algorithm_forward(
                batch_X=ind_X, target=ind_attr_labels.nonzero().view(-1)
            )  # [K, 1, H, W]
            concepts_attribution = concepts_attribution.sum(dim=1, keepdim=True)
            concepts_attribution = (
                concepts_attribution
                - concepts_attribution.amin(dim=(1, 2, 3))[:, None, None, None]
            ) / (
                concepts_attribution.amax(dim=(1, 2, 3))
                - concepts_attribution.amin(dim=(1, 2, 3))
            )[:, None, None, None]  # [K, 1, H, W]

            masked_image = concepts_attribution * ind_X  # [K, C, H, W]
            avg_drop, avg_inc, avg_gain = concepts_adi(
                model.encode_as_concepts,
                ind_X,
                masked_image,
                ind_attr_labels.unsqueeze(0),
            )
            avg_drop_list.append(avg_drop)
            avg_inc_list.append(avg_inc)
            avg_gain_list.append(avg_gain)

    return (
        torch.stack(avg_drop_list).mean(),
        torch.stack(avg_inc_list).mean(),
        torch.stack(avg_gain_list).mean(),
    )


def concepts_adi(
    model: Callable,
    images: torch.Tensor,
    masked_images: torch.Tensor,
    active_labels: torch.Tensor,
):
    """
    Args:
        model: nn.Module
        images: [1, 3, H, W]
        masked_images: [K, 3, H, W]
        active_labels: [1, D] where sum(active_labels) == K

        where K -> active concept label, D -> concept label

    """
    images = images
    masked_images = masked_images

    concepts_logits = None
    masked_concepts_logits = None
    with torch.no_grad():
        concepts_logits: torch.Tensor = model(images)  # [1, D]
        masked_concepts_logits: torch.Tensor = model(masked_images)  # [B, D]

    Y = concepts_logits[:, active_labels[0].bool()].sigmoid()  # [1, D] -> [K, ]
    O = masked_concepts_logits[:, active_labels[0].bool()].sigmoid()  # [K, D] -> [K, ]

    avg_drop = torch.maximum(Y - O, torch.zeros_like(Y)) / Y  # [K, ]
    avg_gain = torch.maximum(O - Y, torch.zeros_like(Y)) / (1 - Y)  # [K, ]
    avg_inc = torch.gt(O, Y)  # [K, ]

    return avg_drop.mean(), avg_inc.float().mean(), avg_gain.mean()


def estimate_top_concepts_accuracy(concept_predictions, concept_labels):
    bs = concept_predictions.shape[0]
    mini_batch_correct_concepts = 0
    mini_batch_total_concepts = 0

    for i in range(bs):
        k_val = int(torch.sum(concept_labels[i]).item())  # active labels for that class
        _, top_gt_indices = torch.topk(concept_labels[i], k=k_val, dim=-1)
        _, top_pred_indices = torch.topk(concept_predictions[i], k=k_val, dim=-1)

        for k in top_pred_indices:
            mini_batch_total_concepts += 1
            if k in top_gt_indices:
                mini_batch_correct_concepts += 1

    return mini_batch_correct_concepts, mini_batch_total_concepts


def val_one_epoch(val_data_loader: DataLoader, model: CBM_Net, device: torch.device):
    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, data in enumerate(val_data_loader):
            images, class_labels, concept_labels, use_concept_labels = (
                None,
                None,
                None,
                None,
            )
            if isinstance(data, list):
                if data.__len__() == 3:
                    images, class_labels, concept_labels = data
                    if isinstance(concept_labels, list):
                        concept_labels = (
                            torch.stack(concept_labels)
                            .permute((1, 0))
                            .to(class_labels.device)
                        )
                else:
                    images, class_labels, concept_labels, use_concept_labels = data
            else:
                images, class_labels, concept_labels = (
                    data["img"],
                    data["og_class_label"],
                    data["attr_labels"],
                )

            # Loading data and labels to device
            images = images.squeeze().to(device)
            class_labels = class_labels.squeeze().to(device)
            concept_labels = concept_labels.squeeze().to(device)
            if use_concept_labels is not None:
                use_concept_labels = use_concept_labels.squeeze().to(device)

            if class_labels.size().__len__() == 2:
                class_labels = torch.reshape(
                    class_labels, (class_labels.shape[0] * 2, 1)
                ).squeeze()
                concept_labels = torch.reshape(
                    concept_labels, (concept_labels.shape[0] * 2, concept_labels.shape[-1])
                )
                if use_concept_labels is not None:
                    use_concept_labels = torch.reshape(
                        use_concept_labels, (use_concept_labels.shape[0] * 2, 1)
                    ).squeeze()

            # Forward
            class_predictions, concept_predictions, _ = model(images)

            # calculate acc per minibatch
            sum_correct_pred += (
                (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
            )
            total_samples += len(class_labels)
            mbcc, mbtc = estimate_top_concepts_accuracy(
                concept_predictions, concept_labels
            )
            concept_acc += mbcc
            concept_count += mbtc

    acc = round(sum_correct_pred / total_samples, 4) * 100
    total_concept_acc = round(concept_acc / concept_count, 4) * 100
    return acc, total_concept_acc


def eval_model_explainability(
    args,
    model: CBM_Net,
    preprocess: transforms.Compose,
    dataset: dataset_collection,
    concept_bank: ConceptBank,
    explain_method: str,
):
    explain_algorithm: GradientAttribution = getattr(
        model_explain_algorithm_factory, explain_method
    )(forward_func=model.encode_as_concepts, model=model)
    explain_algorithm_forward: Callable = getattr(
        model_explain_algorithm_forward, explain_method
    )
    explain_algorithm_forward = partial(
        explain_algorithm_forward, explain_algorithm=explain_algorithm
    )

    eval_save_to = os.path.join(args.save_path, "evaluations")
    if not os.path.exists(eval_save_to):
        os.mkdir(eval_save_to)

    # eval_attribution_alignment(
    #     args,
    #     preprocess,
    #     model,
    #     concept_bank,
    #     explain_algorithm_forward,
    #     eval_save_to,
    # )

    eval_explain_method(args, model, dataset.test_loader, explain_algorithm_forward, eval_save_to)


def eval_attribution_alignment(
    args,
    preprocess: transforms.Compose,
    model: CBM_Net,
    concept_bank: ConceptBank,
    explain_algorithm_forward: Callable,
    eval_save_to: str,
):
    model.eval()
    if "rival10" in args.dataset:
        rival10_dataset = load_dataset(
            dataset_configure(
                dataset="rival10_full",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            ),
            preprocess,
        )

        explain_concept: torch.Tensor = torch.arange(
            0, concept_bank.concept_info.concept_names.__len__()
        ).to(args.device)

        # Start Rival attrbution alignment evaluation
        (
            concepts_segmentation_metric,
            classes_segmentation_metric,
            weighted_cocnepts_metric,
        ) = interpret_all_concept(
            args,
            model,
            rival10_dataset.test_loader,
            explain_algorithm_forward,
            explain_concept,
        )

        args.logger.info(
            concepts_segmentation_metric.format_output(
                RIVAL10_constants._ALL_CLASSNAMES,
                concept_bank.concept_info.concept_names,
            )
        )
        args.logger.info(
            weighted_cocnepts_metric.format_output(
                RIVAL10_constants._ALL_CLASSNAMES,
                concept_bank.concept_info.concept_names,
            )
        )
        args.logger.info(
            classes_segmentation_metric.format_output(RIVAL10_constants._ALL_CLASSNAMES)
        )

        torch.save(
            concepts_segmentation_metric.format_dict(),
            os.path.join(eval_save_to, "concepts_segmentation_metric.pt"),
        )
        torch.save(
            weighted_cocnepts_metric.format_dict(),
            os.path.join(eval_save_to, "weighted_cocnepts_metric.pt"),
        )
        torch.save(
            classes_segmentation_metric.format_dict(),
            os.path.join(eval_save_to, "classes_segmentation_metric.pt"),
        )
    # torch.save(saliency_map_pack, os.path.join(eval_save_to, "saliency_map_pack.pt"))


def eval_explain_method(
    args, model: CBM_Net, data_loader: DataLoader, explain_algorithm_forward: Callable, eval_save_to: str,
):
    avg_drop, avg_inc, avg_gain = compute_adi(
        args,
        model,
        data_loader,
        explain_algorithm_forward,
    )
    args.logger.info(f"avg_drop: {avg_drop * 100:.2f}\\%, avg_inc: {avg_inc * 100:.2f}\\%, avg_gain: {avg_gain * 100:.2f}\\%")
    torch.save({
            "avg_drop": avg_drop,
            "avg_inc" : avg_inc,
            "avg_gain": avg_gain,
        }, os.path.join(eval_save_to, "adi_pack.pt"))
    
    return avg_drop, avg_inc, avg_gain
