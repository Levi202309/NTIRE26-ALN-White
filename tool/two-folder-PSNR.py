"""
Author: Baoney
Time: 2025/7/5 15:28
coding: UTF-8
"""
import os
from PIL import Image
import numpy as np

# 指定两个文件夹路径
folder1 = r'C:\Users\29396\Desktop\26AAAI\experiments\Outputs\0705\Infer-LB-NAFNet-TtoV-VAE-resoN_20250705_155500\visualization'   # 文件夹A路径
folder2 = r'C:\Users\29396\Desktop\26AAAI\experiments\Outputs\0622\Infer-pre-VQVAE-hso-256x256-10-0623-4\test'   # 文件夹B路径
# folder2 = r'C:\Users\29396\Desktop\26AAAI\experiments\Outputs\0705\Infer-LB-NAFNet-TtoV-VAE-resoN_20250705_141945\visualization'   # 文件夹B路径

# 找出两个文件夹中都有的文件名
files1 = set(f for f in os.listdir(folder1) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp')))
files2 = set(f for f in os.listdir(folder2) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp')))
common = sorted(list(files1 & files2))

if not common:
    print("没有找到匹配的图片文件。")
    exit()

psnrs = []
for fname in common:
    img1 = Image.open(os.path.join(folder1, fname)).convert('RGB')
    img2 = Image.open(os.path.join(folder2, fname)).convert('RGB')
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BICUBIC)
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    print(f"{fname}: PSNR = {psnr:.3f} dB")
    psnrs.append(psnr)

print(f"\n总共比对{len(psnrs)}张，平均PSNR: {np.mean(psnrs):.3f} dB")
