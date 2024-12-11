bash _interpret_base.sh "/home/ksas/uzuki_space/adv-cbm/outputs/clip_rival10_asgt/adversarial_saliency_guided_training-open_clip:RN50.pth" \
"open_clip:RN50" \
"/home/ksas/uzuki_space/adv-cbm/data/ckpt/RIVAL_10_asgt/pcbm_rival10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt" \
"layer_grad_cam" "$1" "$2" "$3"