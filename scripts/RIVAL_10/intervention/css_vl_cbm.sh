bash scripts/RIVAL_10/intervention/_attr_align_eval_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"concept_banks/multimodal_concept_open_clip_ViT-B-16_rival10.pkl" \
"css_pcbm" \
"outputs/css_vl_cbm_train_simple_concepts/css_cbm_open_clip:ViT-B-16.pt" \
"layer_grad_cam_vit" \
"CSS_VL_CBM" \
"--batch-size=8"