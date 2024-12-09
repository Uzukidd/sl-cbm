cd ../../..
python attr_align_evalutaion.py --dataset="rival10"\
            --backbone-ckpt="$1"\
            --backbone-name="$2"\
            --concept-bank="/home/ksas/Public/datasets/rival10_concept_bank/multimodal_concept_clip:RN50_rival10_recurse:1.pkl"\
            --pcbm-ckpt="data/ckpt/RIVAL_10/pcbm_RIVAL10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt"\
            --explain-method="$3"\
            --universal-seed="24"\
            $4