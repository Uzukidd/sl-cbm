import argparse
import csv
import os
from tqdm import tqdm

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", required=True, type=str)

    return parser.parse_args()

def main(args):
    csv_file_path = args.imgs_path.rstrip('/\\') + ".csv"
    print(csv_file_path)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',',
                lineterminator='\n',
                quotechar = "'"
                )
        writer.writerow(["filepath", "caption"])
        
        for filename in tqdm(os.listdir(args.imgs_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(args.imgs_path, filename)
                parts = filename.split('_')
                class_name = parts[1].split('.')[0]
                writer.writerow([filepath,f'"A picture of {class_name}"'])

    # try:
    #     # 以读取模式打开文件
    #     with open(csv_file_path, 'r', encoding='utf-8') as file:
    #         # 读取所有行
    #         content = file.readlines()
            
    #     print(' '.join(content)) 

    # except FileNotFoundError:
    #     print(f"文件未找到: {csv_file_path}")
    # except Exception as e:
    #     print(f"发生错误: {e}")
if __name__ == "__main__":
    args = config()
    main(args)