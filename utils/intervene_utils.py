import random
from .model_utils import *
from .explain_utils import concepts_adi


def calculate_c_score(
    criterion: str,
    attr_preds_sigmoid: torch.Tensor,
    attr_labels: torch.Tensor,
    model: CBM_Net,
):
    if criterion == "lcp":
        score = torch.abs(attr_preds_sigmoid - attr_labels) ** 2
    elif criterion == "ucp":
        score = 1 / torch.abs(attr_preds_sigmoid - 0.5) ** 2
    elif criterion == "cctp":
        # if "weight" in .state_dict():
        #     layer_name = model2.weight
        # else:
        #     layer_name = model2.linear.weight
        score = torch.sum(torch.abs(model.nec_concepts_projection.weight), axis=0)
        # # if args.batch_intervention:  # tile up weights in batch intervention
        # #     n_samples = int(len(attr_labels) / args.n_attributes)
        score = score * attr_preds_sigmoid

    return score


def calculate_intervention_order(
    criterion,
    model,
    input_X,
    attr_preds_sigmoid,
    attr_labels,
    concepts_count,
    batch_intervention,
    explain_algorithm_forward: Callable,
    n_trials=1,
):

    B, C, W, H = input_X.size()
    all_intervention_order_list = (
        []
    )  # n_trials * N_TEST * n_groups (order of intervention)
    for i in range(n_trials):
        if criterion in ["ectp", "eudtp"]:
            # if group_intervention:
            #     n_orders = args.n_groups
            # else:
            #     n_orders = args.n_attributes
            # score_list = np.zeros((len(b_class_labels), n_orders)) # intervene on increasing order of score
            # for j in range(n_orders):
            #     if group_intervention:
            #         score = np.zeros(len(b_class_labels))
            #         for attr_idx in GROUP_DICT[j]:
            #             attr_score = calculate_etp_score(criterion, b_class_logits, np.full(len(b_class_labels), attr_idx), ptl_5, ptl_95, model2, b_attr_outputs, b_attr_outputs_sigmoid, b_class_labels, use_relu, use_sigmoid, connect_CY=False)
            #             score += attr_score
            #         score = score/len(GROUP_DICT[j])
            #     else:
            #         score = calculate_etp_score(criterion, b_class_logits, np.full(len(b_class_labels), j), ptl_5, ptl_95, model2, b_attr_outputs, b_attr_outputs_sigmoid, b_class_labels, use_relu, use_sigmoid, connect_CY=False)
            #     score_list[:, j] = score
            # if batch_intervention:
            #     score_list = score_list.flatten()
            #     intervention_order_list = np.argsort(score_list)
            # else:
            #     intervention_order_list = np.argsort(score_list, axis=1)
            # all_intervention_order_list.append(intervention_order_list)
            raise NotImplementedError
        elif criterion in ["ag"]:
            _, K = attr_labels.size()
            intervention_order_list = []
            for batch_mask in range(B):
                ind_X = input_X[batch_mask : batch_mask + 1]  # [1, C, H, W]
                ind_attr_labels = attr_labels[batch_mask]

                model.zero_grad()
                all_explain_concept: torch.Tensor = torch.arange(
                    0, ind_attr_labels.size(-1)
                ).to(ind_attr_labels.device)

                all_concepts_attribution: torch.Tensor = explain_algorithm_forward(
                    batch_X=ind_X, target=all_explain_concept
                )
                all_concepts_attribution = all_concepts_attribution.sum(
                    dim=1, keepdim=True
                )  # torch.Size([18, 1, 224, 224])

                all_concepts_attribution = (
                    all_concepts_attribution
                    - all_concepts_attribution.amin(dim=(1, 2, 3))[:, None, None, None]
                ) / (
                    all_concepts_attribution.amax(dim=(1, 2, 3))
                    - all_concepts_attribution.amin(dim=(1, 2, 3))
                )[
                    :, None, None, None
                ]  # [K, 1, H, W]

                masked_image = (
                    all_concepts_attribution.expand(
                        [
                            all_concepts_attribution.size(0),
                            3,
                            all_concepts_attribution.size(2),
                            all_concepts_attribution.size(3),
                        ]
                    )
                    * ind_X
                )  # [K, C, H, W]
                avg_drop, avg_inc, avg_gain = concepts_adi(
                    model.encode_as_concepts,
                    ind_X,
                    masked_image,
                    None,
                    reduction=None,
                )
                avg_gain = torch.gather(avg_gain, 1, all_explain_concept.unsqueeze(1))
                intervention_order_list.append(-avg_gain.view(-1))
            intervention_order_list = torch.argsort(torch.stack(intervention_order_list))
        else:  # 'rand', 'ucp', 'lcp', 'cctp'
            if batch_intervention:
                if criterion == "rand":
                    # if group_intervention:
                    #     whole_size = args.n_groups * len(b_class_labels)
                    # else:
                    #     whole_size = args.n_attributes * len(b_class_labels)
                    # intervention_order_list = list(random.sample(list(range(whole_size)), whole_size))
                    intervention_order_list = torch.stack(
                        [torch.randperm(concepts_count) for _ in range(B)], dim=0
                    )
                else:
                    score_list = calculate_c_score(
                        criterion, attr_preds_sigmoid, attr_labels, model
                    )
                    # if group_intervention:
                    #     score = []
                    #     for img_id in range(len(b_class_labels)):
                    #         for group_idx in range(args.n_groups):
                    #             group_score = 0
                    #             for attr_idx in GROUP_DICT[group_idx]:
                    #                 group_score += score_list[
                    #                     img_id * args.n_attributes + attr_idx
                    #                 ]
                    #             score.append(group_score / len(GROUP_DICT[group_idx]))
                    # else:
                    #     score = score_list
                    intervention_order_list = torch.argsort(score_list)
                    # import pdb;pdb.set_trace()
                    # raise NotImplementedError
            else:  # single intervention
                raise NotImplementedError
                # intervention_order_list = [] # N_TEST * n_groups (order of intervention)
                # for img_id in range(len(b_class_labels)):
                #     attr_preds = b_attr_outputs[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                #     attr_preds_sigmoid = b_attr_outputs_sigmoid[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                #     attr_preds2 = b_attr_outputs2[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                #     attr_labels = b_attr_labels[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                #     if group_intervention:
                #         n_orders = args.n_groups
                #     else:
                #         n_orders = args.n_attributes
                #     if criterion == 'rand':
                #         intervention_order = list(random.sample(list(range(n_orders)), n_orders))
                #     else:
                #         score_list = calculate_c_score(criterion, attr_preds_sigmoid, attr_labels, model2)
                #         if group_intervention:
                #             score = []
                #             for group_idx in range(args.n_groups):
                #                 group_score = 0
                #                 for attr_idx in GROUP_DICT[group_idx]:
                #                     group_score += score_list[attr_idx]
                #                 score.append(group_score/len(GROUP_DICT[group_idx]))
                #         else:
                #             score = score_list
                #         intervention_order = np.argsort(score)[::-1]
                #     intervention_order_list.append(intervention_order)
        all_intervention_order_list.append(intervention_order_list)
    return all_intervention_order_list
