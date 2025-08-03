python -m pdb -c continue css_vl_cbm_train.py --backbone-name="resnet18_cub" \
    --dataset="css_cub" \
    --target-dataset="spss_cub" \
    --backbone-ckpt="model_zoo/resnet_cub" \
    --concept-bank="concept_banks/cub_resnet18_cub_0.1_100.pkl"\
    --universal-seed="24" \
    --epoch="5" \
    --explain-method="layer_grad_cam" \
    --intervention \
    --exp-name="css_resnet_cbm_train_new"
