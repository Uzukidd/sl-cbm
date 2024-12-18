cd ../../..
python model_robust_training.py --dataset="rival10"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --backbone-name="open_clip:RN50"\
            --concept-bank="/home/ksas/Public/datasets/rival10_concept_bank/multimodal_concept_clip:RN50_rival10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/RIVAL_10/pcbm_rival10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="saliency_map"\
            --train-method="adversarial_saliency_guided_training"\
            --batch-size="64"\
            --exp-name="clip_rival10_asgt"\
            --universal-seed="24"\
            # --k="3e-1"\
            $1