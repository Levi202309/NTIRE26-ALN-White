import os
# 用来产生训练数据配对图像索引txt文件：

# folder1_path = '/root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/train_800/blended/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/train_800/transmission_layer/'
# output_file_path = '/root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/ntire_SIRR25.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire_24_sh_rem_final_test_inp'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/testing_NAFNet-w23Dw24Dw1920P-23eval25.11-On24test/'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_24.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/lighting/AMBIEN6K_test/in'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/lighting/AMBIEN6K_test/gt'
# output_file_path = '/root/autodl-tmp/data/lighting/AMBIEN6K.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/lighting/crop/data-ALN-in-infer-wIFBlend-pre/in-2304x1728'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/lighting/crop/gt-2304x1728'
# output_file_path = '/root/autodl-tmp/data/lighting/ALN25-2304x1728.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/GenerationMask_on_eval25.11-On24test/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/testing_NAFNet-w23Dw24Dw1920P-23eval25.11-On24test/'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_shadow_24.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/lighting/in/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/lighting/gt/'
# output_file_path = '/root/autodl-tmp/data/lighting/ntire25.txt'# 生成txt文件的路径


# folder1_path = '/root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/'
# output_file_path = '/root/autodl-tmp/data/shadow/ntire2024_val.txt'# 生成txt文件的路径


# folder1_path = '/root/autodl-tmp/data/shadow/NTIRE23_sr_val_inp/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/ntire23_sr_valid_gt/'
# output_file_path = '/root/autodl-tmp/data/shadow/ntire2023_val.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire23_sr_train_input/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/ntire23_sr_train_gt/'
# output_file_path = '/root/autodl-tmp/data/shadow/ntire2023.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire24_shrem_train_inp/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/ntire24_shrem_train_gt/'
# output_file_path = '/root/autodl-tmp/data/shadow/ntire2024.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire2025_sh_rem_train/in/'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/ntire2025_sh_rem_train/gt/'
# output_file_path = '/root/autodl-tmp/data/shadow/ntire2025.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/lighting/AMBIEN6K_test/infer-AMBIEN6K-test-wIFBlend-pre/in'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/lighting/AMBIEN6K_test/gt'
# output_file_path = '/root/autodl-tmp/data/lighting/Inf-AMBIEN6K-test.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire_24_sh_rem_final_test_inp'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/testing_NAFNet-train25-test24'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_24_train25.txt'# 生成txt文件的路径
 
# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_25_vit'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_running_result_25_vit.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/NPR/train/inp-pre'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/NPR/train/gt-sony'
# output_file_path = '/root/autodl-tmp/data/NPR/NPR-train.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/NPR/val1/inp-pre'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/NPR/val1/gt-sony'
# output_file_path = '/root/autodl-tmp/data/NPR/NPR-val1.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/NPR/val1/inp-pre-2048'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/NPR/val1/gt-sony-2048'
# output_file_path = '/root/autodl-tmp/data/NPR/NPR-val1-2048.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_25_shadowR'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_running_result_25_shadowR.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/25_fake_shadow_1'     # 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_25_naf'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_shadow_25_naf_1.txt'       # 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/25_fake_shadow_1_R'     # 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_25_shadowR'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_shadow_25_shadowR_1.txt'       # 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/lighting/Test_inputs/ntire25_aln_test_in_match+pseudo2/in'     # 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/lighting/Test_inputs/ntire25_aln_test_in_match+pseudo2/gt'
# output_file_path = '/root/autodl-tmp/data/lighting/ntire25_aln_test_in_match+pseudo2.txt'       # 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/w.oInputEnsemble-L20_shadowR'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_25_shadowR_finetune.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp_stage1'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/fake_lable_25_my'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_lable_25_my_stage1.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp_stage1'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_25_naf'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_running_result_25_naf_stage1.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp_stage1'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_25_shadowR'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_running_result_25_shadowR_stage1.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/shadow/ntire_24_sh_rem_final_test_inp_stage1'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/running_result_24_shadowR'
# output_file_path = '/root/autodl-tmp/data/shadow/fake_label_24_shadowR_stage1.txt'# 生成txt文件的路径


# folder1_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_compare_in_stage1'# 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_compare_gt'
# output_file_path = '/root/autodl-tmp/data/shadow/ntire25_sh_rem_test_compared_stage1.txt'# 生成txt文件的路径

# folder1_path = '/root/autodl-tmp/data/lighting/Test_inputs/ntire25_aln_test_in'     # 两个文件夹的路径
# folder2_path =  '/root/autodl-tmp/data/lighting/RefineData/RFData-ntire25-aln-test-in-match+pseudo2/gt'
# output_file_path = '/root/autodl-tmp/data/lighting/ntire25_aln_test_in_match+pseudo2.txt'       # 生成txt文件的路径

folder1_path = '/root/autodl-tmp/Dataset/white/ntire26_aln_test_in'     # 两个文件夹的路径
folder2_path =  '/root/autodl-tmp/Dataset/white/ntire26white_nafnet_best_test_pseudo/w.oInputEnsemble'
output_file_path = '/root/autodl-tmp/Dataset/white/26_white_test_pseudo_NAFNetBest.txt'       # 生成txt文件的路径




# 获取两个文件夹中的文件列表
folder1_files = sorted(os.listdir(folder1_path))
folder2_files = sorted(os.listdir(folder2_path))

# 确保两个文件夹中的文件数量相同
if len(folder1_files) != len(folder2_files):
    raise ValueError("len(folder1_files) != len(folder2_files)")
assert len(folder1_files) == len(folder2_files)


# 写入文件路径到txt文件中
with open(output_file_path, 'w') as f:
    for file1, file2 in zip(folder1_files, folder2_files):
        # 不去除共同的根路径版本
        # f.write(os.path.join(folder1_path, file1) + ' ' + os.path.join(folder2_path, file2) + '\n')  
        # 去除共同的根路径版本
        file1_path = os.path.join(folder1_path, file1)
        file2_path = os.path.join(folder2_path, file2)
        # 去掉共同的根路径，并写入到txt文件中
        f.write(file1_path.replace('/root/autodl-tmp/', '') + ' ' + file2_path.replace('/root/autodl-tmp/', '') + '\n')

print(f"已生成配对图像路径的txt文件：{output_file_path}")
