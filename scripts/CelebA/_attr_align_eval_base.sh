python attr_align_evalutaion.py --target-dataset="smiling_celebA_test"\
            --dataset="smiling_celebA_test"\
            --backbone-ckpt="$1"\
            --backbone-name="$2"\
            --concept-bank="$3"\
            --pcbm-arch="$4"\
            --pcbm-ckpt="$5"\
            --explain-method="$6"\
            --universal-seed="24"\
            --exp-name="$7"\
            $8

