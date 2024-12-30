cd ../../..
python ls_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
    --backbone-ckpt="laion2b-s34b-b88K" \
    --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
    --universal-seed="24" \
    --exp-name="ls_vl_cbm_train_simple_concepts" \
    --not-save-ckpt

# python css_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --universal-seed="24" \
#     --exp-name="css_vl_cbm_train_simple_concepts" \
#     --pcbm-ckpt="outputs/css_vl_cbm_train_simple_concepts/css_cbm_open_clip:ViT-B-16.pt" \
#     --evaluate