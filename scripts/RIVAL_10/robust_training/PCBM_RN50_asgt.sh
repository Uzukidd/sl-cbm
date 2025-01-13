cd ../../..
python model_robust_training.py --dataset="rival10"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --backbone-name="open_clip:RN50"\
            --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_clip:RN50_rival10.pkl"\
            --pcbm-ckpt="data/ckpt/RIVAL_10/pcbm_rival10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="saliency_map"\
            --train-method="adversarial_saliency_guided_training"\
            --pcbm-arch="robust_pcbm"\
            --batch-size="32"\
            --exp-name="PCBM_RN50_RIVAL10_ASGT"\
            --universal-seed="24"\
            $1