bash _interpret_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
"css_pcbm" \
"/home/ksas/uzuki_space/adv-cbm/outputs/css_vl_cbm_train_simple_concepts/css_cbm_open_clip:ViT-B-16.pt" \
"layer_grad_cam_vit" "$1" "$2" "$3"