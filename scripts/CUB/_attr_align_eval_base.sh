# cd ../..
python attr_align_evalutaion.py --dataset=$8\
            --target-dataset=$8\
            --backbone-ckpt="$1"\
            --backbone-name="$2"\
            --concept-bank="$3"\
            --pcbm-arch="$4"\
            --pcbm-ckpt="$5"\
            --explain-method="$6"\
            --universal-seed="24"\
            --exp-name="$7"\
            $9
            # --intervention\
            # $9

