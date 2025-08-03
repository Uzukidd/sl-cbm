# bash scripts/CUB/_attr_align_eval_base.sh "model_zoo/resnet_cub" \
# "resnet18_cub" \
# "concept_banks/cub_resnet18_cub_0.1_100.pkl" \
# "pcbm" \
# "data/ckpt/CUB/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt" \
# "layer_grad_cam" \
# "VIS-PCBM-RN18" \
# "spss_cub" \
# "--batch-size=8 --batch-vis --device=1"

bash scripts/CUB/_attr_align_eval_base.sh "model_zoo/resnet_cub" \
"resnet18_cub" \
"concept_banks/cub_resnet18_cub_0.1_100.pkl" \
"css_pcbm" \
"data/ckpt/CUB/css_resnet_cbm_train/css_cbm_resnet18_cub.pt" \
"layer_grad_cam" \
"VIS-CSS-PCBM-RN18" \
"css_cub" \
"--batch-size=8 --batch-vis --device=1"

bash scripts/CUB/_attr_align_eval_base.sh "model_zoo/resnet_cub" \
"resnet18_cub" \
"concept_banks/cub_resnet18_cub_0.1_100.pkl" \
"spss_pcbm" \
"data/ckpt/CUB/spss_resnet_cbm_TCAV_1.0_1.0_1.0/trainable_weights.pt" \
"layer_grad_cam" \
"VIS-SPSS-PCBM-RN18" \
"spsss_cub" \
"--batch-size=8 --batch-vis --device=1"