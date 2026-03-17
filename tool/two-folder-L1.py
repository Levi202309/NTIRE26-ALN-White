"""
Author: Baoney
Time: 2025/7/5 14:35
coding: UTF-8
"""
import os
from PIL import Image
import numpy as np

# 指定比较的两个文件夹
folder1 = r'C:\Users\29396\Desktop\26AAAI\experiments\Outputs\0622\Infer-pre-VQVAE-hso-256x256-10-0623-4\test'   # 文件夹A路径
# folder2 = r'C:\Users\29396\Desktop\26AAAI\experiments\Outputs\0622\Infer-pre-VQVAE-hso-256x256-10-0623-4\test'   # 文件夹B路径
folder2 = r'C:\Users\29396\Desktop\26AAAI\experiments\Outputs\0705\Infer-pre-VQVAE3-val-hso-1120x640-num_levels3\test'   # 文件夹B路径

# 找出两个文件夹中都有的文件名
files1 = set(f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
files2 = set(f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
common = sorted(list(files1 & files2))

if not common:
    print("没有找到匹配的图片文件。")
    exit()

all_loss = []
for fname in common:
    img1 = Image.open(os.path.join(folder1, fname)).convert('RGB')
    img2 = Image.open(os.path.join(folder2, fname)).convert('RGB')

    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BICUBIC)

    arr1 = np.array(img1).astype(np.float32) / 255.
    arr2 = np.array(img2).astype(np.float32) / 255.

    l1 = np.mean(np.abs(arr1 - arr2))
    print(f"{fname}: L1 Loss = {l1:.6f}")
    all_loss.append(l1)

print(f"\n总共比对{len(all_loss)}张，平均L1损失: {np.mean(all_loss):.6f}")
