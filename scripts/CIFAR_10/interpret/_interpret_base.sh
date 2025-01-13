cd ../../..
python concept_interpretability.py --dataset="cifar10"\
            --backbone-ckpt="/home/ksas/Public/model_zoo/clip"\
            --backbone-name="$1"\
            --concept-bank="$6"\
            --explain-method="$2"\
            --universal-seed="24"\
            --class-target="$3"\
            --concept-target="$4"\
            $5\
            --pcbm-ckpt="$7"