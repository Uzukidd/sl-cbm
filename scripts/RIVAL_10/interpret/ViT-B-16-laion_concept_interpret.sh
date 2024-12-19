bash _interpret_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
"pcbm" \
"data/ckpt/RIVAL_10/pcbm_rival10__open_clip:ViT-B-16__multimodal_concept_open_clip:ViT-B-16_rival10__lam:0.0002__alpha:0.99__seed:42.ckpt" \
"layer_grad_cam_vit" "$1" "$2" "$3"