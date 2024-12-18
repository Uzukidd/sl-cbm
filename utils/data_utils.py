import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm

from .constants import *

# UPDATE _DATA_ROOT to '{path to dir where rival10.zip is unzipped}/RIVAL10/'
# _DATA_ROOT = "/home/ksas/Public/datasets/RIVAL10/"
# _LABEL_MAPPINGS = './datasets/label_mappings.json'
# _WNID_TO_CLASS = './datasets/wnid_to_class.json'

# _ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
#               'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy', 
#               'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']

def attr_to_idx(attr):
    return RIVAL10_features._ALL_ATTRS.index(attr)

def idx_to_attr(idx):
    return RIVAL10_features._ALL_ATTRS[idx]

def resize(img): 
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def save_uri_as_img(uri, fpath='tmp.png'):
    ''' saves raw mask and returns it as an image'''
    import base64
    from io import BytesIO
    image_data = base64.b64decode(uri)
    image_data = BytesIO(image_data)
    img = mpimg.imread(image_data, format='jpg')

    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img

class LocalRIVAL10(Dataset):
    def __init__(self, train=True, classification_output=True, masks_dict=True, transform=None):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.

        See __getitem__ for more documentation. 
        '''
        self.train = train
        self.data_root = dataset_constants.RIVAL10_DIR.format('train' if self.train else 'test')
        self.classification_output = classification_output
        self.masks_dict = masks_dict
        self.transform = transform

        self.instance_types = ['ordinary']
        # NOTE: 
        # if include_aug:
        #     self.instance_types += ['superimposed', 'removed']
        
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224,224))

        with open(RIVAL10_features._LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(RIVAL10_features._WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            for f in tqdm(glob.glob(dir_path+'/*')):
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    # def transform(self, imgs):
    #     transformed_imgs = []
    #     i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
    #     coin_flip = (random.random() < 0.5)
    #     for ind, img in enumerate(imgs):
    #         if self.train:
    #             img = TF.crop(img, i, j, h, w)

    #             if coin_flip:
    #                 img = TF.hflip(img)

    #         img = TF.to_tensor(self.resize(img))
            
    #         if img.shape[0] == 1:
    #             img = torch.cat([img, img, img], axis=0)
            
    #         if self.normalizer is not None:
    #             img = self.normalizer(img)
            
    #         transformed_imgs.append(img)

    #     return transformed_imgs

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224,224,3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path,  merged_mask_path, mask_dict_path = self.all_instances[i]

        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long() # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        imgs = [img, merged_mask_img]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                mask_dict = dict()

            for attr in mask_dict:
                mask_uri = mask_dict[attr]
                mask = save_uri_as_img(mask_uri)
                imgs.append(Image.fromarray(np.uint8(255*mask)))

        transformed_imgs = [self.transform(img) for img in imgs]
        img = transformed_imgs.pop(0)
        merged_mask = transformed_imgs.pop(0)
        out = dict({'img':img, 
                    'attr_labels': attr_labels, 
                    'changed_attrs': changed_attrs,
                    'merged_mask' :merged_mask,
                    'og_class_name': class_name,
                    'og_class_label': class_label})

        if self.classification_output:
            return out["img"], out["og_class_label"]

        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(RIVAL10_features._ALL_ATTRS)+1)]
            for i, attr in enumerate(mask_dict):
                # if attr == 'entire-object':
                ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                attr_masks[ind] = transformed_imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)

        return out
    
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
                                      classification_output = False, 
                                      transform=transform)
          self._parse_train_test(self.dataset)

        elif split == "test":
          self.dataset = LocalRIVAL10(train=False, 
                                      masks_dict=False, 
                                      classification_output = False, 
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