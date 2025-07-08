bash scripts/CIFAR_10/attr_align_eval/_attr_align_eval_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"concept_banks/multimodal_concept_open_clip_ViT-B-16_rival10.pkl" \
"pcbm" \
"data/ckpt/RIVAL_10/RIVAL_10/pcbm_rival10__open_clip_ViT-B-16__multimodal_concept_open_clip_ViT-B-16_rival10__lam_0.0002__alpha_0.99__seed_42.ckpt" \
"layer_grad_cam_vit" \
"PCBM-ViT-B-16-laion" \
"--batch-size=8"