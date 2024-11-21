cd ../../..
python embedding_robust_training.py --dataset="$1"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --backbone-name="$2"\
            --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="$3"\
            --train-method="$4"\
            --regular="$5"\
            --batch-size="16"\
            --exp-name="$1_$2_$3_$4_$5"\
            --universal-seed="24"\
            $6