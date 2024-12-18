class CLIP_constants:
    pass

class dataset_constants:
    CIFAR10_DIR:str = "/home/ksas/Public/datasets/cifar10_concept_bank"
    CIFAR100_DIR:str = "/home/ksas/Public/datasets/cifar100_concept_bank"
    CUB_DATA_DIR:str = "/home/ksas/Public/datasets/CUB_DATASET/CUB_200_2011"
    CUB_PROCESSED_DIR:str = "/home/ksas/Public/datasets/CUB_DATASET/class_attr_data_10"
    RIVAL10_DIR:str = "/home/ksas/Public/datasets/RIVAL10/{}/"

class CUB_features:
    # body part = (min, max)
    has_bill_shape = (0, 8)
    has_wing_color = (9, 23)
    has_upperparts_color = (24, 38)
    has_underparts_color = (39, 53)
    has_breast_pattern = (54, 57)
    has_back_color = (58, 72)
    has_tail_shape = (73, 78)
    has_upper_tail_color = (79, 93)
    has_head_pattern = (94, 104)
    has_breast_color = (105, 119)
    
    has_throat_color = (120, 134)
    has_eye_color = (135, 148)
    has_bill_length = (149, 151)
    has_forehead_color = (152, 166)
    has_under_tail_color = (167, 181)
    has_nape_color = (182, 196)
    has_belly_color = (197, 211)
    has_wing_shape = (212, 216)
    has_size = (217, 221)
    has_shape = (222, 235)
    has_back_pattern = (236, 239)
    has_tail_pattern = (240, 243)
    has_belly_pattern = (244, 247)
    has_primary_color = (248, 262)
    has_leg_color = (263, 277)
    has_bill_color = (278, 292)
    has_crown_color = (293, 307)
    has_wing_pattern = (308, 311)
    
    part_mask = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

class model_zoo:
    CLIP:str = "/home/ksas/Public/model_zoo/clip"

class RIVAL10_features:
    _LABEL_MAPPINGS = './utils/RIVAL10/label_mappings.json'
    _WNID_TO_CLASS = './utils/RIVAL10/wnid_to_class.json'

    _ALL_CLASSNAMES = ["truck", "car", "plane", "ship", "cat", "dog", "equine", "deer", "frog", "bird"]

    _ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
                'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy', 
                'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']
    
    _ZERO_SHOT_ATTRS = [
    'an animal with long-snout', 
    'an animal with  wings', 
    'a vehicle with wheels', 
    'has text written on it', 
    'an animal with  horns', 
    'an animal with floppy-ears', 
    'an animal with ears', 
    'an animal with colored-eyes', 
    'an object or an animal with a tail', 
    'an animal with mane', 
    'an animal with beak', 
    'an animal with hairy coat', 
    'an object with a metallic body', 
    'an object with rectangular shape', 
    'is damp, wet, or watery ', 
    'a long object', 
    'a tall object', 
    'has patterns on it'
    ]