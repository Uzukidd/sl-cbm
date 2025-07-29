
run_eval() {
python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
    --backbone-ckpt="laion2b-s34b-b88K" \
    --concept-bank="concept_banks/multimodal_concept_open_clip_ViT-B-16_rival10.pkl" \
    --pcbm-ckpt="outputs/lambda_ablation_5e-1/$1/trainable_weights.pt" \
    --universal-seed="24" \
    --exp-name="eval/lambda_ablation_5e-1/$1" \
    --not-save-ckpt \
    --explain-method="builtin_explain" \
    --evaluate \
    --device="cuda:1" \
    $5
}

for file in outputs/lambda_ablation_5e-1/*; do
    filename=$(basename "$file")
    run_eval "$filename"
done
# python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --pcbm-ckpt="outputs/lambda_ablation/trainable_weights.pt" \
#     --universal-seed="24" \
#     --exp-name="lambda_ablation/eval_spss_vl_cbm_train_simple_concepts_$1_$2_$3_$4" \
#     --not-save-ckpt \
#     --explain-method="builtin_explain" \
#     --evaluate \
#     --device="cuda:1" \
#     $5