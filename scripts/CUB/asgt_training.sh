cd ../..
python model_robust_training.py --dataset="cub"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/resnet_cub"\
            --backbone-name="resnet18_cub"\
            --concept-bank="/home/ksas/Public/datasets/cub_concept_bank/cub_resnet18_cub_0.1_100.pkl"\
            --pcbm-ckpt="data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="saliency_map"\
            --train-method="adversarial_saliency_guided_training"\
            --batch-size="64"\
            --exp-name="resnet18_cub_adversarial_saliency_guided_training"\
            --universal-seed="24"\
            $1