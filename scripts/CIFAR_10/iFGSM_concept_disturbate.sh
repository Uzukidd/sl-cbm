cd ../..
python concept_disturbation.py --dataset="cifar10"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --backbone-name="clip:RN50"\
            --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --cocnept-selecting-func="weights_concept_selecting"\
            --attack-func="iFGSM"\
            --epsilon=$1