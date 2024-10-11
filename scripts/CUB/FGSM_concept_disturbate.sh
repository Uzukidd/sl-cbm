cd ../..
python concept_disturbation.py --dataset="cub"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/resnet_cub"\
            --backbone-name="resnet18_cub"\
            --concept-bank="/home/ksas/Public/datasets/cub_concept_bank/cub_resnet18_cub_0.1_100.pkl"\
            --pcbm-ckpt="data/ckpt/CUB/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --cocnept-selecting-func="weights_concept_selecting"\
            --attack-func="FGSM"\
            --epsilon=$1