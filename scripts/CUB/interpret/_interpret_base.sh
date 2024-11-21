cd ../../..
python concept_interpretability.py --dataset="cub"\
            --backbone-ckpt="$1"\
            --backbone-name="resnet18_cub"\
            --concept-bank="/home/ksas/Public/datasets/cub_concept_bank/cub_resnet18_cub_0.1_100.pkl"\
            --pcbm-ckpt="$2"\
            --explain-method="$3"\
            --universal-seed="24"\
            --class-target="$4"\
            --concept-target="$5"\
            $6
