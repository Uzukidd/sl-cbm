cd ../../..
python concept_interpretability.py --dataset="$1"\
            --backbone-ckpt="$2"\
            --backbone-name="$3"\
            --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="$4"\
            --universal-seed="24"\
            --class-target="$5"\
            --concept-target="$6"\
            $7