bash _attr_align_eval_base.sh "outputs/PCBM_ViT16_RIVAL10_ASGT/robust_pcbm_open_clip:ViT-B-16.pt" \
"open_clip:ViT-B-16" \
"/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
"robust_pcbm" \
"data/ckpt/RIVAL_10/pcbm_rival10__open_clip:ViT-B-16__multimodal_concept_open_clip:ViT-B-16_rival10__lam:0.0002__alpha:0.99__seed:42.ckpt" \
"layer_grad_cam_vit" \
"robust-PCBM-ViT16" \
"--batch-size=8"