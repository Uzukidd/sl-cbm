cd ../../..
python concept_interpretability.py --dataset="rival10"\
            --backbone-ckpt="$1"\
            --backbone-name="$2"\
            --concept-bank="/home/ksas/Public/datasets/rival10_concept_bank/multimodal_concept_clip:RN50_rival10_recurse:1.pkl"\
            --pcbm-ckpt="$3"\
            --explain-method="$4"\
            --universal-seed="24"\
            --class-target="$5"\
            --concept-target="$6"\
            $7