import torch
import clip
from utils import *

backbone = load_backbone(backbone_configure(
            backbone_name="clip:RN50",
            backbone_ckpt="model_zoo/clip",
            device=torch.device(0),
        ))

rival10_dataset = load_dataset(dataset_configure(
            dataset="rival10",
            batch_size=64,
            num_workers=64,
        ), backbone.preprocess)

cifar10_dataset = load_dataset(dataset_configure(
            dataset="cifar10",
            batch_size=64,
            num_workers=64,
        ), backbone.preprocess)

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

text_tokens = clip.tokenize(cifar10_classes).to(torch.device(0))
text_features = backbone.backbone_model.encode_text(text_tokens)
text_features = text_features
correct = 0
total = 0
from tqdm import tqdm
# 遍历测试集
for images, labels in tqdm(rival10_dataset.test_loader):
    images = images.to(torch.device(0))
    images = backbone.normalizer(images)
    
    image_features = backbone.backbone_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 相似度
    logits_per_image = image_features @ text_features.T
    preds = logits_per_image.argmax(dim=-1)

    correct += (preds.cpu() == labels).sum().item()
    total += labels.size(0)


print(f"Zero-shot accuracy on one batch: {correct / total:.2%}")

# 遍历测试集
correct = 0
total = 0
for images, labels in tqdm(cifar10_dataset.test_loader):
    images = images.to(torch.device(0))
    images = backbone.normalizer(images)
    
    image_features = backbone.backbone_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 相似度
    logits_per_image = image_features @ text_features.T
    preds = logits_per_image.argmax(dim=-1)

    correct += (preds.cpu() == labels).sum().item()
    total += labels.size(0)


print(f"Zero-shot accuracy on one batch: {correct / total:.2%}")