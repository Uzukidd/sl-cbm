bash scripts/CelebA/_attr_align_eval_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"concept_banks/multimodal_concept_open_clip:ViT-B-16_smiling_celebA.pkl" \
"pcbm" \
"data/ckpt/celebA/pcbm_smiling_celebA__open_clip:ViT-B-16__multimodal_concept_open_clip:ViT-B-16_smiling_celebA__lam:0.0002__alpha:0.99__seed:42.ckpt" \
"layer_grad_cam_vit" \
"PCBM-ViT-B-16-laion" \
"--batch-size=32"