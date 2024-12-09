cd ../../..
python concept_interpretability.py --dataset="cifar10"\
            --backbone-ckpt="data/ckpt/CIFAR_10/robust_open_clip:RN50_k:0.3.pth"\
            --backbone-name="open_clip:RN50"\
            --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="layer_grad_cam"\
            --universal-seed="24"\
            --class-target="$1"\
            --concept-target="$2"\
            $3