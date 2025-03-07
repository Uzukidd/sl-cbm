cd ../../..
python -m pdb -c continue css_vl_cbm_train.py --backbone-name="resnet18_cub" \
    --dataset="css_cub" \
    --backbone-ckpt="/home/ksas/Public/model_zoo/resnet_cub" \
    --concept-bank="/home/ksas/Public/datasets/concept_banks/cub_resnet18_cub_0.1_100.pkl"\
    --universal-seed="24" \
    --explain-method="layer_grad_cam" \
    --exp-name="css_resnet_cbm_train" \
    --not-save-ckpt
