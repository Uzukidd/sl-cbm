import rival10
from rival10 import LocalRIVAL10
from utils.constants import dataset_constants

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader, Dataset, Sampler

from tqdm import tqdm

rival10.constants.RIVAL10_constants.set_rival10_dir(dataset_constants.RIVAL10_DIR)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

trainset = LocalRIVAL10(train=True, cherrypick_list = None, masks_dict=True, transform=transform)
testset = LocalRIVAL10(train=False, cherrypick_list = None, masks_dict=True, transform=transform)

class_to_idx = {c: i for (i,c) in enumerate(rival10.constants.RIVAL10_constants._ALL_CLASSNAMES)}
idx_to_class = {v: k for k, v in class_to_idx.items()}
train_loader = DataLoader(trainset, batch_size=64,
                            shuffle=False, num_workers=4)
test_loader = DataLoader(testset, batch_size=64,
                            shuffle=False, num_workers=4)

def verify_attr_labels_consistency(data_set):
    # 用字典存储每个类别第一次出现的 attr_labels
    class_to_attr = {}

    for idx, data in enumerate(data_set):
        cls = data["og_class_label"]
        attr = data["attr_labels"]

        # 如果第一次遇到这个类别，存下来
        if cls not in class_to_attr:
            class_to_attr[cls] = attr
        else:
            # 后续遇到这个类别，就和第一次的 attr_labels 做比较
            if not torch.equal(attr, class_to_attr[cls]):
                print(class_to_attr[cls])
                print(attr)
                print(f"❌ Inconsistent attr_labels found for class {cls} at index {idx}")
                return False

    print("✅ All attr_labels are consistent for each og_class_label")
    return True

def compute_attr_intersection(data_set):
    class_to_attrs = {}

    for data in tqdm(data_set):
        cls = data["og_class_label"]
        attr = data["attr_labels"]

        if cls not in class_to_attrs:
            # 第一次出现，先赋值
            class_to_attrs[cls] = attr.clone()  # 用 clone() 避免原地修改
        else:
            # 后续遇到同一类别，做按位与（交集）
            class_to_attrs[cls] = class_to_attrs[cls] | attr

    return class_to_attrs

result = compute_attr_intersection(trainset)

for cls, attrs in result.items():
    print(f"Class {cls}: {attrs.tolist()}")
    
# Class 4: [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
# Class 1: [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# Class 0: [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# Class 9: [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
# Class 6: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Class 5: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Class 2: [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# Class 7: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
# Class 8: [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
# Class 3: [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# verify_attr_labels_consistency(trainset)
# data_demo = trainset[0]
# # print(data_demo)
# print(data_demo.keys())
# print(data_demo["og_class_label"])
# print(data_demo["attr_labels"])