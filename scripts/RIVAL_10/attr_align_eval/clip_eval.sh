bash _attr_align_eval_base.sh "/home/ksas/Public/model_zoo/clip" \
"clip:RN50" \
"/home/ksas/Public/datasets/concept_banks/multimodal_concept_clip:RN50_rival10.pkl" \
"pcbm" \
"data/ckpt/RIVAL_10/pcbm_rival10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt" \
"layer_grad_cam" \
"PCBM-RN50" \
"$1"