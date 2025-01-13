cd ../../..
python model_robust_training.py --dataset="rival10"\
            --backbone-ckpt="laion2b-s34b-b88K"\
            --backbone-name="open_clip:ViT-B-16"\
            --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl"\
            --pcbm-ckpt="data/ckpt/RIVAL_10/pcbm_rival10__open_clip:ViT-B-16__multimodal_concept_open_clip:ViT-B-16_rival10__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="saliency_map"\
            --train-method="adversarial_saliency_guided_training"\
            --pcbm-arch="robust_pcbm"\
            --batch-size="32"\
            --exp-name="PCBM_ViT16_RIVAL10_ASGT"\
            --universal-seed="24"\
            $1