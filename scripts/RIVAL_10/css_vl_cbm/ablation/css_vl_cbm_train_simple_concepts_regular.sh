cd ../../../..
K=("0.1" "0.3" "0.5" "0.7")
loss4_scale=("0.1" "1.0" "10.0" "100.0" "1000.0")
for i in "${K[@]}"; do
    for j in "${loss4_scale[@]}"; do
        echo "K = $i, loss4_scale = $j"
        python css_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
            --backbone-ckpt="laion2b-s34b-b88K" \
            --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
            --universal-seed="24" \
            --explain-method="layer_grad_cam_vit" \
            --cross-entropy-regular \
            --k="$i" \
            --loss4-scale="$j" \
            --not-save-ckpt \
            --exp-name="css_regular_ablation_layergrad_ori_explain/K:$i-loss4_scale:$j"
    done
done
# python css_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --universal-seed="24" \
#     --explain-method="saliency_map" \
#     --cross-entropy-regular \
#     --K==
#     --not-save-ckpt \
#     --exp-name="css_vl_cbm_train_simple_concepts_regular" \

# python css_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
#     --backbone-ckpt="laion2b-s34b-b88K" \
#     --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
#     --universal-seed="24" \
#     --exp-name="css_vl_cbm_train_simple_concepts" \
#     --pcbm-ckpt="outputs/css_vl_cbm_train_simple_concepts/css_cbm_open_clip:ViT-B-16.pt" \
#     --evaluate