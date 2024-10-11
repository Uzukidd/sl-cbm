cd ../..
python concept_disturbation.py --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --cocnept-selecting-func="weights_concept_selecting"\
            --attack-func="FGSM"\
            --epsilon=$1
# python concept_disturbation.py --dataset-path="/home/ksas/Public/datasets/cifar10_concept_bank"\
#             --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
#             --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
#             --pcbm-ckpt="data/ckpt/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
#             --cocnept-selecting-func="randomK_concept_selecting"\
#             --attack-func="FGSM"

# python concept_disturbation.py --dataset-path="/home/ksas/Public/datasets/cifar10_concept_bank"\
#             --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
#             --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
#             --pcbm-ckpt="data/ckpt/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
#             --cocnept-selecting-func="leastK_concept_selecting"\
#             --attack-func="FGSM"

# python concept_disturbation.py --dataset-path="/home/ksas/Public/datasets/cifar10_concept_bank"\
#             --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
#             --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
#             --pcbm-ckpt="data/ckpt/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
#             --cocnept-selecting-func="topK_concept_selecting"\
#             --attack-func="FGSM"