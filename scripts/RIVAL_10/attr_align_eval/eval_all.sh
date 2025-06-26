# bash _attr_align_eval_base.sh "/home/ksas/Public/model_zoo/clip" \
# "clip:RN50" \
# "/home/ksas/Public/datasets/concept_banks/multimodal_concept_clip:RN50_rival10.pkl" \
# "pcbm" \
# "data/ckpt/RIVAL_10/pcbm_rival10__clip:RN50__multimodal_concept_clip:RN50_rival10_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt" \
# "layer_grad_cam" \
# "VIS-PCBM-RN50" \
# "--batch-size=8 --batch-vis"

# bash _attr_align_eval_base.sh "laion2b-s34b-b88K" \
# "open_clip:ViT-B-16" \
# "/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
# "pcbm" \
# "data/ckpt/RIVAL_10/pcbm_rival10__open_clip:ViT-B-16__multimodal_concept_open_clip:ViT-B-16_rival10__lam:0.0002__alpha:0.99__seed:42.ckpt" \
# "layer_grad_cam_vit" \
# "VIS-PCBM-ViT-B-16-laion" \
# "--batch-size=8 --batch-vis"

bash _attr_align_eval_base.sh "laion2b-s34b-b88K" \
"open_clip:ViT-B-16" \
"/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
"css_pcbm" \
"outputs/css_vl_cbm_train_simple_concepts/css_cbm_open_clip:ViT-B-16.pt" \
"layer_grad_cam_vit" \
"VIS-CSS-PCBM-ViT-B-16-laion" \
"--batch-size=8 --batch-vis"

# bash _attr_align_eval_base.sh "laion2b-s34b-b88K" \
# "open_clip:ViT-B-16" \
# "/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
# "css_pcbm" \
# "outputs/css_vl_cbm_train_simple_concepts/css_cbm_open_clip:ViT-B-16.pt" \
# "layer_grad_cam_vit" \
# "CSS-PCBM-ViT-B-16-laion" \
# "--batch-size=8"