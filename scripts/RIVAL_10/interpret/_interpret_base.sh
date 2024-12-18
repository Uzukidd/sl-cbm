cd ../../..
python concept_interpretability.py --dataset="rival10"\
            --backbone-ckpt="$1"\
            --backbone-name="$2"\
            --concept-bank="$3"\
            --pcbm-arch="$4"\
            --pcbm-ckpt="$5"\
            --explain-method="$6"\
            --universal-seed="24"\
            --class-target="$7"\
            --concept-target="$8"\
            $9