cd /home/ksas/uzuki_space/open_clip/src
python -m open_clip_train.main \
    --val-data="/home/ksas/Public/datasets/cifar10_concept_bank/training_imgs.csv"  \
    --model="RN50" \
    --csv-caption-key="caption" \
    --csv-img-key="filepath" \
    --csv-separator="," \
    --pretrained="/home/ksas/uzuki_space/open_clip/src/logs/2024_10_23-22_22_39-model_RN50-lr_1e-05-b_128-j_3-p_amp/checkpoints/epoch_2.pt"