cd ../..
python training_tools/learn_concepts_dataset.py --dataset-name="cub"\
            --backbone-ckpt="data/ckpt/CUB/robust_resnet18_cub.pth"\
            --C 0.001 0.01 0.1 1.0 10.0\
            --n-samples=100\
            --out-dir="data/concept_bank/CUB"