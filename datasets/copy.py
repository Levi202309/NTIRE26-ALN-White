import os
import shutil

def copy_images_with_suffix(src_dir, dst_dir, suffix="_in", start_num=0, count=50):
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    
    for i in range(start_num, start_num + count):
        # 生成带后缀的文件名 (e.g. 00000_in.png)
        # filename = f"{i:05d}{suffix}.png"
        filename = f"{i:04d}.png"
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        # 检查源文件是否存在
        if not os.path.exists(src_path):
            print(f"警告: 文件 {filename} 不存在，已跳过")
            continue
            
        shutil.copy2(src_path, dst_path)
        copied += 1
        print(f"已复制: {filename} ({copied}/{count})")
    
    print(f"\n操作完成，成功复制 {copied} 个文件 (失败 {count - copied} 个)")

if __name__ == "__main__":
    # 修改以下路径
    source_folder = "/root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt"
    target_folder = "/root/autodl-tmp/data/shadow/24/gt"
    
    copy_images_with_suffix(source_folder, target_folder, suffix="_gt")
