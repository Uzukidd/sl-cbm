cd /home/ksas/uzuki_space/open_clip/src
python -m open_clip_train.main \
    --dataset-type="csv" \
    --train-data="/home/ksas/Public/datasets/cifar10_concept_bank/training_imgs.csv" \
    --warmup=1000 \
    --batch-size=128 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs=2 \
    --workers=3 \
    --model="ViT-B-32" \
    --report-to="wandb" \
    --csv-caption-key="caption" \
    --csv-img-key="filepath" \
    --csv-separator="," \
    --pretrained="/home/ksas/Public/model_zoo/clip/ViT-B-32.pt" \
    --log-every-n-steps=100 \
    --coca-contrastive-loss-weight=0 \
    --coca-caption-loss-weight=1