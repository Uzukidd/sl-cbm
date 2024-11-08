cd ../../..
python model_robust_training.py --dataset="cifar10"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --backbone-name="open_clip:RN50"\
            --concept-bank="/home/ksas/Public/datasets/cifar10_concept_bank/multimodal_concept_clip:RN50_cifar10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/CIFAR_10/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="saliency_map"\
            --train-method="adversarial_saliency_guided_training"\
            --batch-size="64"\
            --exp-name="clip_adversarial_saliency_guided_training"\
            --universal-seed="24"\
            $1