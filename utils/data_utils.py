import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import glob
import json
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm

from pcbm.data.cub import load_cub_data
from rival10 import LocalRIVAL10

from .constants import *
from typing import Callable, Union, Optional

class CSS_CUB_Dataset(Dataset):
    def __init__(self, split, true_batch_size, pkl_paths, transform:Optional[transforms.Compose], percentage_of_concept_labels_for_training = 0.3):

        # every call returns a batch of image pairs (strictly set batch_size=1 in DataLoader of train script)
        # this makes sure that every pair in a batch belongs to a different class (to satisfy contrastive loss)
        self.batch_size = true_batch_size 
        # sort all samples in a classwise fashion
        self.classwise_samples = []
        for _ in range(200): self.classwise_samples.append([])
        
        if split == "train":
            trainset, _ = load_cub_data([pkl_paths], use_attr=True, no_img=False, 
                batch_size=1, uncertain_label=False, image_dir=dataset_constants.CUB_DATA_DIR, resol=224, normalizer=None,
                n_classes=200, resampling=False)
            self.data_len = trainset.__len__()
            self._parse_train_test(trainset)
            self.data = trainset
          
        elif split == "test":
          testset, _ = load_cub_data([pkl_paths], use_attr=True, no_img=False, 
                batch_size=1, uncertain_label=False, image_dir=dataset_constants.CUB_DATA_DIR, resol=224, normalizer=None,
                n_classes=200, resampling=False)
          self.data_len = testset.__len__()
          self._parse_train_test(testset)
          self.data = testset
          
        self.transform = transform

        # use concept labels for only 30% of the samples
        self.concept_labelled_samples = []
        for i in np.random.choice(np.arange(self.data_len),size=int(percentage_of_concept_labels_for_training * self.data_len),replace=False): self.concept_labelled_samples.append(i)

    # sort all samples in a classwise fashion
    def _parse_train_test(self, anno_list):
        for idx, data in enumerate(anno_list):
            self.classwise_samples[data[1]].append(idx)
    
    # value of __len__ is chosen such that each sample is approximately seen only once per epoch
    def __len__(self):
        return int(self.data_len/(self.batch_size*2))

    def __getitem__(self, dummy_idx):

        # first choose unique classes of size = batch_size = 64
        classes = np.random.choice(np.arange(200),size=self.batch_size,replace=False)
        # will take the shape [batch_size,2_imgs,3,224,224]
        image_pairs = []
        # will take the shape [batch_size,2_identical_labels]
        label_pairs = []
        # will take the shape [batch_size,2,31]
        concept_label_pairs = []
        # will take the shape [batch_size,2 bool values]
        use_concept_labels = []

        for i,cls_num in enumerate(classes):
            image_pairs.append([])
            label_pairs.append([])
            concept_label_pairs.append([])
            use_concept_labels.append([])

            # choose two random samples from same class
            sample_index = np.random.choice(np.arange(len(self.classwise_samples[cls_num])), size=2, replace=False)

            for idx in sample_index:
                secondary_idx = self.classwise_samples[cls_num][idx]
                img_data = self.data[secondary_idx]
                img = img_data[0]
                
                if self.transform:
                    img = self.transform(img)

                image_pairs[i].append(img)
                label_pairs[i].append(torch.tensor(img_data[1]))

                concept_label = img_data[2]
                
                concept_label_pairs[i].append(torch.tensor(concept_label))
        
                if idx in self.concept_labelled_samples:
                    use_concept_labels[i].append(torch.tensor(1))
                else:
                    use_concept_labels[i].append(torch.tensor(0))
            
            # shape=(2,3,224,224)
            image_pairs[i] = torch.stack(image_pairs[i],0)
            label_pairs[i] = torch.stack(label_pairs[i],0)
            concept_label_pairs[i] = torch.stack(concept_label_pairs[i],0)
            use_concept_labels[i] = torch.stack(use_concept_labels[i],0)
        
        image_pairs = torch.stack(image_pairs,0)
        label_pairs = torch.stack(label_pairs,0)
        concept_label_pairs = torch.stack(concept_label_pairs,0)
        use_concept_labels = torch.stack(use_concept_labels,0)

        return image_pairs, label_pairs, concept_label_pairs, use_concept_labels
    
class CSS_Rival_Dataset(Dataset):
    def __init__(self, split, true_batch_size, percentage_of_concept_labels_for_training, transform=None):

        # every call returns a batch of image pairs (strictly set batch_size=1 in DataLoader of train script)
        # this makes sure that every pair in a batch belongs to a different class (to satisfy contrastive loss)
        self.batch_size = true_batch_size 
        # sort all samples in a classwise fashion
        self.classwise_samples = []
        for _ in range(10): self.classwise_samples.append([])

        if split == "train":
          self.dataset = LocalRIVAL10(train=True, 
                                      masks_dict=False, 
                                      transform=transform)
          self._parse_train_test(self.dataset)

        elif split == "test":
          self.dataset = LocalRIVAL10(train=False, 
                                      masks_dict=False,  
                                      transform=transform)
          self._parse_train_test(self.dataset)

        # rival10 dataset class already performs: image -> tensor -> random_resized_crop(train split only) -> random_horizontal_flip(train split only) -> resize to (224,224)

        # use concept/attribute labels for X % of samples
        self.concept_labelled_samples = []
        for i in np.random.choice(np.arange(len(self.dataset)),size=int(percentage_of_concept_labels_for_training * len(self.dataset)),replace=False): self.concept_labelled_samples.append(i)

    # sort all samples in a classwise fashion
    def _parse_train_test(self, _dataset):
        for i in range(len(_dataset)):
            cls_label = _dataset[i]['og_class_label']
            self.classwise_samples[cls_label].append(i)
    
    def __len__(self):
        return int(len(self.dataset)/(self.batch_size*2))

    def __getitem__(self, dummy_idx):
        
        # first choose unique classes of size = batch_size
        classes = np.random.choice(np.arange(10),size=self.batch_size,replace=True)
        # will take the shape [batch_size,2_imgs,3,224,224]
        image_pairs = []
        # will take the shape [batch_size,2_identical_labels]
        label_pairs = []
        # will take the shape [batch_size,2,18]
        concept_label_pairs = []
        # will take the shape [batch_size,2 bool values]
        use_concept_labels = []

        for i,cls_num in enumerate(classes):
            image_pairs.append([])
            label_pairs.append([])
            concept_label_pairs.append([])
            use_concept_labels.append([])

            # choose two random samples from same class
            sample_index = np.random.choice(np.asarray(self.classwise_samples[cls_num]), size=2, replace=False)

            for idx in sample_index:

                (img, attr_labels, cls_label) = (
                        self.dataset[idx]['img'], 
                        self.dataset[idx]['attr_labels'], 
                        self.dataset[idx]['og_class_label']
                        )

                image_pairs[i].append(img)

                if cls_num != cls_label:
                    print(cls_num,cls_label)
                    assert cls_num == cls_label
                label_pairs[i].append(torch.tensor(cls_num))

                attr_labels = attr_labels.type(torch.float32)
                concept_label_pairs[i].append(attr_labels)
        
                if idx in self.concept_labelled_samples:
                    use_concept_labels[i].append(torch.tensor(1.))
                else:
                    use_concept_labels[i].append(torch.tensor(0.))
            
            # shape=(2,3,224,224)
            image_pairs[i] = torch.stack(image_pairs[i],0)
            label_pairs[i] = torch.stack(label_pairs[i],0)
            concept_label_pairs[i] = torch.stack(concept_label_pairs[i],0)
            use_concept_labels[i] = torch.stack(use_concept_labels[i],0)
        
        image_pairs = torch.stack(image_pairs,0)
        label_pairs = torch.stack(label_pairs,0)
        concept_label_pairs = torch.stack(concept_label_pairs,0)
        use_concept_labels = torch.stack(use_concept_labels,0)

        return image_pairs, label_pairs, concept_label_pairs, use_concept_labels
    