cd ../../..
python concept_interpretability.py --dataset="cub"\
            --backbone-ckpt="/home/ksas/uzuki_space/adv-cbm/data/ckpt/CUB/robust_resnet18_cub.pth"\
            --backbone-name="resnet18_cub"\
            --concept-bank="data/concept_bank/CUB/cub_resnet18_cub_0.1_100.pkl"\
            --pcbm-ckpt="data/ckpt/CUB/robust_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="integrated_gradient"\
            --universal-seed="24"\
            --class-target="$1"\
            --concept-target="$2"\
            $3