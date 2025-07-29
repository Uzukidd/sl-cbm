import numpy as np
from rival10 import RIVAL10_constants

class CLIP_constants:
    pass

class dataset_constants:
    image_size:np.ndarray = np.array([3, 244, 244])
    CIFAR10_DIR:str = "data/CIFAR10"
    CIFAR100_DIR:str = "data/CIFAR100"
    CUB_DATA_DIR:str = "data/CUB_DATASET/CUB_200_2011"
    CUB_PROCESSED_DIR:str = "data/CUB_DATASET/class_attr_data_10"
    CELEBA_DIR:str = "data/"
    RIVAL10_DIR:str = "data/RIVAL10"

class celebA_features:
    text_attributions = ["5 o'Clock Shadow",
        "Arched Eyebrows",
        "Attractive",
        "Bags Under Eyes",
        "Bald",
        "Bangs",
        "Big Lips",
        "Big Nose",
        "Black Hair",
        "Blond Hair",
        "Blurry",
        "Brown Hair",
        "Bushy Eyebrows",
        "Chubby",
        "Double Chin",
        "Eyeglasses",
        "Goatee",
        "Gray Hair",
        "Heavy Makeup",
        "High Cheekbones",
        "Male",
        "Mouth Slightly Open",
        "Mustache",
        "Narrow Eyes",
        "No Beard",
        "Oval Face",
        "Pale Skin",
        "Pointy Nose",
        "Receding Hairline",
        "Rosy Cheeks",
        "Sideburns",
        "Smiling",
        "Straight Hair",
        "Wavy Hair",
        "Wearing Earrings",
        "Wearing Hat",
        "Wearing Lipstick",
        "Wearing Necklace",
        "Wearing Necktie",
        "Young"]
    
    smiling_concepts = [
        "Bags Under Eyes",
        "High Cheekbones",
        "Mouth Slightly Open",
        "Rosy Cheeks",
        "Double Chin",
        "Arched Eyebrows",
        "Narrow Eyes",
    ]
    
    smiling_concepts_indices = [3, 19, 21, 29, 14, 1, 23]
    
    smiling_label_index = 31


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

RIVAL10_constants.label_concepts_upperbound = {
    4: [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    1: [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    0: [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    9: [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
    6: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    5: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    2: [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    7: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    8: [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    3: [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
}

RIVAL10_constants.label_cifar10_to_rival10 = {
    0: 2,  # airplane → plane
    1: 1,  # automobile → car
    2: 9,  # bird → bird
    3: 4,  # cat → cat
    4: 7,  # deer → deer
    5: 5,  # dog → dog
    6: 8,  # frog → frog
    7: 6,  # horse → equine
    8: 3,  # ship → ship
    9: 0,  # truck → truck
}

class model_zoo:
    CLIP:str = "model_zoo/clip"