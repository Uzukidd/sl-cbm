cd ../../..
python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
    --backbone-ckpt="laion2b-s34b-b88K" \
    --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
    --universal-seed="24" \
    --exp-name="spss_vl_cbm_train_simple_concepts" \
    --explain-method="layer_grad_cam_vit" \
    --not-save-ckpt

# python spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --universal-seed="24" \
#     --exp-name="spss_vl_cbm_train_simple_concepts" \
#     --explain-method="builtin_explain" \
#     --not-save-ckpt
