# cd ../../..
python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
    --backbone-ckpt="laion2b-s34b-b88K" \
    --dataset="smiling_celebA" \
    --target-dataset="smiling_celebA_test" \
    --concept-bank="concept_banks/multimodal_concept_open_clip:ViT-B-16_smiling_celebA.pkl" \
    --universal-seed="24" \
    --batch-size="32" \
    --exp-name="lambda_ablation/spss_vl_cbm_train_celebA_concepts_$1_$2_$3" \
    --explain-method="builtin_explain" \
    --lambda1="$1" \
    --lambda2="$2" \
    --lambda3="$3" \
    $4

# python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --universal-seed="24" \
#     --exp-name="spss_vl_cbm_train_simple_concepts" \
#     --explain-method="builtin_explain" \
#     --not-save-ckpt
