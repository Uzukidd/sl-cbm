import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from costants import FGSM_EPS

def main():
    out_data = []
    for eps in tqdm(FGSM_EPS):
        result = subprocess.run([f'bash ../scripts/FGSM_concept_disturbate.sh {eps}'], shell=True, capture_output=True, text=True)
        
        output_dict = {}
        lines = result.stdout.strip().split("\n")
        for line in lines:
            line_split = line.split(":")
            if line_split.__len__() == 2:
                key, value = line.split(":")
                output_dict[key.strip()] = float(value.strip())
        out_data.append(output_dict)


    x = np.array(FGSM_EPS)
    for y_name in out_data[0].keys():
        y = []
        for idx in range(out_data.__len__()):
            y.append(out_data[idx][y_name])
        y = np.array(y)
        plt.clf()
        sns.lineplot(x=x, y=y)
        plt.title(f'FGSM_eps_ablation_plot_{y_name}')
        plt.xlabel('$\\epsilon$')
        plt.ylabel(y_name)

        # 保存图表到本地
        plt.savefig(f'FGSM_eps_ablation_plot_{y_name}.png')
    
if __name__ == "__main__":
    main()