bash scripts/CelebA/_attr_align_eval_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"concept_banks/multimodal_concept_open_clip:ViT-B-16_smiling_celebA.pkl" \
"spss_pcbm" \
"outputs/lambda_ablation/spss_vl_cbm_train_celebA_concepts_1.0_1.0_1.0/trainable_weights.pt" \
"builtin_explain" \
"SPSS-SMILIING-CELEBA" \
"--batch-size=32"