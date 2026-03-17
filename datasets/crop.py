import os
from PIL import Image
from pathlib import Path

# input_folder = '/root/autodl-tmp/data/lighting/in'
input_folder = '/root/autodl-tmp/data/lighting/gt'
target_width = 2304  # 目标宽度
target_height = 1728  # 目标高度

# 创建输出文件夹（格式：test1-1600x1200）
# output_folder = Path(f'/root/autodl-tmp/data/lighting/crop/in-{target_width}x{target_height}')
output_folder = Path(f'/root/autodl-tmp/data/lighting/crop/gt-{target_width}x{target_height}')
output_folder.mkdir(exist_ok=True)
# os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        W, H = img.size

        # 计算裁剪区域（居中裁剪）
        left = (W - target_width) / 2
        top = (H - target_height) / 2
        right = (W + target_width) / 2
        bottom = (H + target_height) / 2

        img_cropped = img.crop((left, top, right, bottom))

        output_filename = os.path.join(output_folder, filename)
        img_cropped.save(output_filename)

        print(f"已处理并保存: {filename}")