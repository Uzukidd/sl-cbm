import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from costants import FGSM_EPS

def main():
    out_data = []
    for explain_method in["layer_grad_cam_concept_interpret",
                          "integrated_grad_concept_interpret"]:
        for concepts in tqdm(["has_bill_shape",
                         "has_wing_color",
                         "has_upperparts_color",
                         "has_underparts_color",
                         "has_breast_pattern",
                         "has_back_color",
                         "has_tail_shape"]):
        
            result = subprocess.run([f'bash {explain_method}.sh "" "{concepts}" "--save-100-local"'], cwd="./scripts/CUB", shell=True, capture_output=True, text=True)
            print(result.stdout.strip().split("\n"))
            print(result.stderr.strip().split("\n"))

    
if __name__ == "__main__":
    main()