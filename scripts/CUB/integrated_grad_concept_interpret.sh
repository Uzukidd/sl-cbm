cd ../..
python concept_interpretability.py --dataset="cub"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/resnet_cub"\
            --backbone-name="resnet18_cub"\
            --concept-bank="/home/ksas/Public/datasets/cub_concept_bank/cub_resnet18_cub_0.1_100.pkl"\
            --pcbm-ckpt="data/ckpt/CUB/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="integrated_gradient"\
            --class-target="$1"\
            --concept-target="$2"

# python concept_interpretability.py --dataset="cifar10"\
#             --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
#             --backbone-name="clip:ViT-B/32"\
#             --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:ViT-B_32_cifar10_recurse:1.pkl"\
#             --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:ViT-B_32__multimodal_concept_clip:ViT-B_32_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
#             --class-target="$1"\
#             --concept-target="$2"
