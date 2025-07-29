bash scripts/CelebA/_attr_align_eval_base.sh "model_zoo/clip" \
"clip:RN50" \
"concept_banks/multimodal_concept_clip:RN50_smiling_celebA.pkl" \
"pcbm" \
"data/ckpt/celebA/pcbm_smiling_celebA__clip:RN50__multimodal_concept_clip:RN50_smiling_celebA__lam:0.0002__alpha:0.99__seed:42.ckpt" \
"layer_grad_cam" \
"PCBM-RN50" \
"--batch-size=32"