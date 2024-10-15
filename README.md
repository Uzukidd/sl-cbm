Install PCBM as a module by running the following command:

# adv-cbm

AdvCBM: Adversarial Disturbation Vs. Concept Bottle Models

## Installation

a. Configure the environment properly from [https://github.com/mertyg/post-hoc-cbm](https://github.com/mertyg/post-hoc-cbm) (Skip if you already did it)

b. Install pcbm as a module by running the following command:

```
pip install git+https://github.com/Uzukidd/pcbm-module
```

c. Use `manually_train.ipynb` to train the model (Checkpoints from [post-hoc-cbm](https://github.com/mertyg/post-hoc-cbm) may not be compatible with pcbm-module)

d. Configure `scripts/CIFAR_10/iFGSM_concept_disturbate.sh` and `class dataset_cosntants` in `common_utils.py` properly

## Disturbation-based method

e. Run an epoch of attack by running the following command:

```
bash scripts/CIFAR_10/iFGSM_concept_disturbate.sh 0.01
```

## IG-based method

e. Run an epoch of attack by running the following command:

```
bash scripts/CIFAR_10/integrated_grad_concept_interpret.sh "airplane" "landing gear"
```
