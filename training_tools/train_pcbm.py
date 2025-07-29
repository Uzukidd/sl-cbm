import argparse
import os
import pickle
import numpy as np
import torch
import time
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import *

from pcbm.models import PosthocLinearCBM
from pcbm.training_tools import load_or_compute_projections


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    
    # PCBM, classifier, cub_net, etc
    parser.add_argument("--pcbm-ckpt", type=str)
    parser.add_argument("--pcbm-arch", default="pcbm", type=str)

    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--dataset-scalar', default=None, type=float)

    return parser.parse_args()


def run_linear_probe(args, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=args.lam, l1_ratio=args.alpha, verbose=0,
                               penalty="elasticnet", max_iter=10000)
    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }
    
    if classifier.coef_.shape[0] > 1:
        coef = classifier.coef_
        intercept = classifier.intercept_
    else:
        coef = np.concatenate([-1 * classifier.coef_, classifier.coef_], axis = 0)
        intercept = np.concatenate([-1 * classifier.intercept_, classifier.intercept_], axis = 0)

    # If it's a binary task, we compute auc
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
    return run_info, coef, intercept



def main(args):
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(dataset.class_to_idx)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, 
                                                                                model, 
                                                                                model.classifier,
                                                                                dataset.train_loader, 
                                                                                dataset.test_loader)
    
    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    
    # Convert from the SGDClassifier module to PCBM module.
    model.set_weights(weights=weights, bias=bias)
    # Sorry for the model path hack. Probably i'll change this later.
    model_path = os.path.join(args.out_dir,
                              f"pcbm_{args.dataset}__{args.backbone_name.replace('/', '_')}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}.ckpt")
    torch.save(model.state_dict(), model_path)

    # Again, a sad hack.. Open to suggestions
    run_info_file = model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    # run_info_file = os.path.join(args.out_dir, run_info_file)
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    
    # if num_classes > 1:
    #     # Prints the Top-5 Concept Weigths for each class.
    #     print(posthoc_layer.analyze_classifier(k=5))

    print(f"Model saved to : {model_path}")
    print(run_info)

if __name__ == "__main__":
    args = config()
    main(args)
