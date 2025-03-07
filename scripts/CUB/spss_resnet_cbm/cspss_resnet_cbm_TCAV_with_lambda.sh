cd ../../..
python -m pdb -c continue spss_vl_cbm_train.py --backbone-name="resnet18_cub" \
    --dataset="css_cub" \
    --backbone-ckpt="/home/ksas/Public/model_zoo/resnet_cub" \
    --concept-bank="/home/ksas/Public/datasets/concept_banks/cub_resnet18_cub_0.1_100.pkl"\
    --universal-seed="24" \
    --exp-name="lambda_ablation/cspss_resnet_cbm_TCAV_$1_$2_$3_$4" \
    --loss="cspss" \
    --explain-method="builtin_explain" \
    --epoch="10" \
    --lambda1="$1" \
    --lambda2="$2" \
    --lambda3="$3" \
    --lambda4="$4" \
    $5

# python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --universal-seed="24" \
#     --exp-name="spss_vl_cbm_train_simple_concepts" \
#     --explain-method="builtin_explain" \
#     --not-save-ckpt
