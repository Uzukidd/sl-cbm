cd ../../..
python -m pdb -c continue spss_vl_cbm_train.py --backbone-name="open_clip:ViT-B-16" \
    --backbone-ckpt="laion2b-s34b-b88K" \
    --concept-bank="/home/ksas/Public/datasets/concept_banks/multimodal_concept_open_clip:ViT-B-16_rival10.pkl" \
    --pcbm-ckpt="outputs/lambda_ablation/cspss_vl_cbm_train_simple_concepts_1.0_1.0_1.0_1.0/trainable_weights.pt" \
    --universal-seed="24" \
    --exp-name="lambda_ablation/eval_spss_vlm_test" \
    --not-save-ckpt \
    --explain-method="builtin_explain" \
    --evaluate \
    --device="cuda:1" \
    $1