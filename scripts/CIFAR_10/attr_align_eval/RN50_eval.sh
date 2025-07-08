bash scripts/CIFAR_10/attr_align_eval/_attr_align_eval_base.sh "model_zoo/clip" \
"clip:RN50" \
"concept_banks/multimodal_concept_clip_RN50_rival10.pkl" \
"pcbm" \
"data/ckpt/RIVAL_10/RIVAL_10/pcbm_rival10__clip_RN50__multimodal_concept_clip_RN50_rival10_recurse_1__lam_0.0002__alpha_0.99__seed_42.ckpt" \
"layer_grad_cam" \
"PCBM-RN50" \
"--batch-size=8"