bash scripts/CIFAR_10/attr_align_eval/_attr_align_eval_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"concept_banks/multimodal_concept_open_clip_ViT-B-16_rival10.pkl" \
"spss_pcbm" \
"outputs/lambda_ablation/spss_vl_cbm_train_simple_concepts_1.0_1.0_5.0/trainable_weights.pt" \
"builtin_explain" \
"SPSS-PCBM-ViT-B-16-laion" \
"$1"