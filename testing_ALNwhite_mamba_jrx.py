import time,torchvision,argparse,logging,sys,os,gc
import torch,random,tqdm,collections
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters,print_args_parameters, compute_ssim
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets_pairs import my_dataset,my_dataset_eval,my_dataset_wTxt
from networks.ECFNet_arch import ECFNet_complete

sys.path.append(os.getcwd())
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,default= "training_SC") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/root/autodl-tmp/result/')

parser.add_argument('--eval_in_path', type=str,default= '/root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp')
parser.add_argument('--eval_gt_path', type=str,default= '/root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt/')
# load load_pre_model
parser.add_argument('--pre_model', type=str, default= '')
parser.add_argument('--model', type=str, default= '')
parser.add_argument('--pre_model_dir', type=str, default= '')

# save results
parser.add_argument('--models_ensemble', type= str2bool, default= False)
parser.add_argument('--inputs_ensemble', type= str2bool, default= False)

# model setting
parser.add_argument('--base_channel', type = int, default= 24)
parser.add_argument('--num_res', type=int, default= 6)
parser.add_argument('--img_channel', type=int, default= 3)
parser.add_argument('--enc_blks', nargs='+', type=int, help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, help='List of integers')
parser.add_argument('--MultiScale', type=str2bool, default= False)   
parser.add_argument('--global_residual', type=str2bool, default= True)  
parser.add_argument('--net_IN', type=str2bool, default= False)  
parser.add_argument('--wfusion', type=str2bool, default= False)  
parser.add_argument('--drop_flag', type=str2bool, default= False)  
parser.add_argument('--drop_rate', type=float, default= 0.1)
parser.add_argument('--activation', type=str, default= 'relu')
parser.add_argument('--kernel_size', type=int, default= 3)

parser.add_argument('--evalD', type=str, default= '')


args = parser.parse_args()
# print all args params!!!
print_args_parameters(args)


exper_name =args.experiment_name



unified_path = args.unified_path
SAVE_PATH =unified_path  + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

    
    
trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])
results_mertircs = SAVE_PATH + exper_name + '.txt'


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 增加计算推理每张图片消耗的时间
def inference(net,eval_loader,Dname = 'S', save_result = False):
    net.eval()
    total_time = 0.0  # 累计总耗时
    count = 0         # 样本计数器

    with torch.no_grad():
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
        # for index, (data_in, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            print(f"✅ 输入尺寸检查: {inputs.shape}")
            
            # 开始计时（确保GPU同步）
            start_time = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()  # 确保CUDA操作完成

            outputs = net(inputs)
            
            # 结束计时（再次同步）
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            # 计算单张耗时
            elapsed = end_time - start_time
            total_time += elapsed
            count += 1
            
            # 打印单张推理时间（保留4位小数）
            print(f"[{name[0]}] 单张推理耗时: {elapsed:.4f}s")
            
            if save_result:
                save_result_path = SAVE_PATH  + '/w.oInputEnsemble/' #+ name[0] + '.png'
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path + name[0] , nrow =1, padding=0 )
        
        # 计算平均耗时（保留4位小数）
        avg_time = total_time / count if count > 0 else 0
        print(f"\n✅ 平均推理时间: {avg_time:.4f}s/image | 总样本数: {count}")

def test_Ifblend(net,eval_loader,Dname = 'S', save_result = False):
    net.eval()
    with torch.no_grad():
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            outputs = net(inputs)
            if save_result:
                save_result_path = SAVE_PATH  + '/w.oInputEnsemble/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)

                                # === 关键修复步骤 ===
                # 1. 转换到 CPU 并解除梯度追踪
                output_tensor = outputs.cpu().detach().squeeze(0)  # [C,H,W]
                
                # 2. 标准化到 [0,255] 并转换类型
                output_np = output_tensor.mul(255).clamp(0, 255).byte().numpy()
                
                # 3. 调整通道顺序为 HWC
                output_np = np.transpose(output_np, (1, 2, 0))  # [H,W,C]
                
                # 4. 使用 PIL 保存（避免 torchvision 的默认压缩）
                from PIL import Image
                img = Image.fromarray(output_np)
                img.save(os.path.join(save_result_path, name[0]))
                # =====================

# mkdir 
# rsync --delete-before -d /root/autodl-tmp/Shadow_Challenge/new_data/ /root/autodl-tmp/Shadow_Challenge/training_NAFNet_0208SC-wRegAug-w23Dw24D_WweightedCharWlongTrain

        
def test_wInputEnsemble(net,eval_loader,Dname = 'S' , save_result = False):
    net.eval()
    with torch.no_grad():
        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            
            normal_outputs = net(inputs)
            
            # flip W 
            input_flipW = torch.flip(inputs, (-1,))
            output_FW = torch.flip(net(input_flipW), (-1,))
            normal_outputs += output_FW
            # flip H
            input_flipH = torch.flip(inputs, (-2,))
            output_FH = torch.flip(net(input_flipH), (-2,))
            normal_outputs += output_FH
            # flip H W
            input_flipHW = torch.flip(inputs, (-2, -1))
            output_FHW = torch.flip(net(input_flipHW), (-2, -1))
            normal_outputs += output_FHW 
            # RGB to GBR
            input_flipHW = inputs[:, [1, 2, 0], :, :]
            output_FHW = net(input_flipHW)[:, [2, 0, 1], :, :]
            normal_outputs += output_FHW 
            
            # final output
            outputs = normal_outputs / 5.0

            
            out_psnr, out_psnr_wClip, out_ssim = compute_psnr(outputs, labels), compute_psnr(torch.clamp(outputs, 0., 1.), labels), compute_ssim(outputs, labels)
            in_psnr, in_ssim = compute_psnr(inputs, labels), compute_ssim(inputs, labels)
            
            Avg_Meters_evaling.update({ 'eval_output_psnr': out_psnr,
                                       'eval_output_psnr_wClip': out_psnr_wClip,
                                        'eval_input_psnr': in_psnr,
                                      'eval_output_ssim': out_ssim,
                                        'eval_input_ssim': in_ssim  })
            content = f'index: {index} | name :{name[0]}|| [in_psnr :{in_psnr}, in_ssim:{in_ssim}|| out_psnr:{out_psnr}, out_psnr_wClip:{out_psnr_wClip} , out_ssim:{out_ssim} ]'
            
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content)
                file.write('\n')
            
            if save_result:
                save_result_path = SAVE_PATH  + '/wInputEnsemble/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path + name[0] , nrow =1, padding=0 )
            
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']

        
        
        #save_imgs_for_visual(save_path, inputs, labels, train_output[-1])
        
        content_ = f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 3)}  / In_SSIM:{round(Final_input_SSIM, 3)}    ||  Out_PSNR:{round(Final_output_PSNR, 3)} | Out_PSNR_wclip:{round(Final_output_PSNR_wclip, 3)}    / OUT_SSIM:{round(Final_output_SSIM, 3)} ]  cost time: { time.time() -st}"
        
        print(content_)
        with open(results_mertircs, 'a') as file:
            file.write(content_)

def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)

def get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path ,trans_eval=trans_eval):
    
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )  # fix_sample 样本限制
    
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers= 4)
    
    return eval_loader

def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))

# 模型集成   
def merge_models( net ,folder_path= args.pre_model_dir):
    #folder_path = '/path/to/your/folder'  # 请将路径替换为您文件夹的实际路径
    model_list = []

    models_names = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
    model_path_list = []
    for i in range(len(models_names)):
        model_path_list.append(folder_path + models_names[i])

    # print the models which need to be merged
    for i in range(len(model_path_list)):
        print('i------:',model_path_list[i])

    # load all pre-trained weights
    for model_path in model_path_list:
        net.eval()
        net.load_state_dict(torch.load(model_path))
        model_list.append(net)
    print("All models loaded successfully")
    print("num of models to be merged is : " + str(len(model_list)))

    merge_model = net
    print("*" * 20)
    print("merging ...")
    worker_state_dict = [x.state_dict() for x in model_list]
    weight_keys = list(worker_state_dict[0].keys()) #tqdm(list(worker_state_dict[0].keys()))
    #print('list(worker_state_dict[0].keys())----------------', list(worker_state_dict[0].keys()))
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        
        #print('key:-------------', key, 'len(model_list)----------', len(model_list))
        #weight_keys.set_description("merging weights %s" % key)
        
        key_sum = 0
        for i in range(len(model_list)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(model_list)
    merge_model.load_state_dict(fed_state_dict)

    return merge_model


# 保存模型和检查点参数名
def save_parameters(model, ckpt_path, save_dir, model_name):
    """
    保存模型和优化器参数的名称、统计信息及张量值
    Args:
        model: 要分析的PyTorch模型
        ckpt_path: 检查点文件路径
        save_dir: 主保存目录
        model_name: 模型标识名称
    """
    # 创建模型参数保存目录
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型参数
    model_params_dir = os.path.join(model_dir, "model_parameters")
    os.makedirs(model_params_dir, exist_ok=True)
    
    # # 模型参数信息文件
    # with open(os.path.join(model_dir, "model_parameters.txt"), "w") as f:
    #     for name, param in model.state_dict().items():
    #         # 保存张量值
    #         tensor_path = os.path.join(model_params_dir, f"{name}.pt")
    #         torch.save(param, tensor_path)
            
    #         # 记录统计信息
    #         f.write(f"Parameter: {name}\n")
    #         f.write(f"Shape: {tuple(param.shape)}\n")
    #         f.write(f"DataType: {param.dtype}\n")
            
    #         if param.numel() > 0:
    #             f.write(f"Mean: {param.mean().item():.6f}\n")
    #             f.write(f"Std: {param.std().item():.6f}\n")
    #             f.write(f"Min: {param.min().item():.6f}\n")
    #             f.write(f"Max: {param.max().item():.6f}\n")
    #         else:
    #             f.write("Empty tensor\n")
    #         f.write(f"Saved to: {tensor_path}\n")
    #         f.write("-" * 80 + "\n")

    # 处理优化器参数
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "optimizer_state_dict" in checkpoint:
        optimizer_dir = os.path.join(model_dir, "optimizer_states")
        os.makedirs(optimizer_dir, exist_ok=True)
        
        # 优化器信息文件
        with open(os.path.join(model_dir, "optimizer_states.txt"), "w") as f:
            optimizer = checkpoint["optimizer_state_dict"]
            
            # 参数组信息
            for i, group in enumerate(optimizer["param_groups"]):
                f.write(f"Parameter Group {i}:\n")
                for key, value in group.items():
                    if key != "params":
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # 参数状态信息
            for param_id, states in optimizer["state"].items():
                f.write(f"Parameter ID: {param_id}\n")
                for state_name, state_tensor in states.items():
                    # 保存状态张量
                    state_path = os.path.join(optimizer_dir, f"param_{param_id}_{state_name}.pt")
                    torch.save(state_tensor, state_path)
                    
                    # 记录状态信息
                    f.write(f"State: {state_name}\n")
                    f.write(f"Shape: {tuple(state_tensor.shape)}\n")
                    f.write(f"DataType: {state_tensor.dtype}\n")
                    
                    if state_tensor.numel() > 0:
                        f.write(f"Mean: {state_tensor.mean().item():.6f}\n")
                        f.write(f"Std: {state_tensor.std().item():.6f}\n")
                        f.write(f"Min: {state_tensor.min().item():.6f}\n")
                        f.write(f"Max: {state_tensor.max().item():.6f}\n")
                    else:
                        f.write("Empty tensor\n")
                    f.write(f"Saved to: {state_path}\n")
                    f.write("-" * 60 + "\n")
                f.write("=" * 80 + "\n")
                
if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=2 python /root/ShadowChallenge/testing_SCwNAFNet.py --experiment_name testing_NAFNet-w23Dw24DwLargeP   --model nafnet   --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6  --pre_model /root/autodl-tmp/Shadow_Challenge/training_NAFNet_0208SC-wRegAug-w23Dw24DwLargeP/net_epoch_70_PSNR_21.73.pth
  
    # 2024/03/20 19:22测试：CUDA_VISIBLE_DEVICES=8 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name   testing_NAFNet_wTLC-w24Dw960P-23eval24.61-On24test --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6 --pre_model_dir /root/autodl-tmp/Shadow_Challenge/temp/ --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/Shadow_Challenge/training_NAFNet_0226SC-wRegAug-w23Dw24DwLargeP960-Arch1-FT-Re23dAdd24test_many318/net_epoch_7_PSNR_24.61.pth --global_residual False  --MultiScale False  --evalD 24-test --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp --patch_size 512 --factor 2

    # 2024/03/21 18:23测试：CUDA_VISIBLE_DEVICES=6 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name   testing_NAFNet_wTLC-w24Dw960P-23eval25.07-On23eval --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6 --pre_model_dir /root/autodl-tmp/Shadow_Challenge/temp/ --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/Shadow_Challenge/Other/training_NAFNet_0226SC-wRegAug-w23Dw24DwLargeP1440-Arch1-FT-Add23d_Add24test_many/net_epoch_7_PSNR_25.07.pth --global_residual False  --MultiScale False  --evalD 23-eval --eval_in_path  /root/autodl-tmp/ShadowDatasets/NTIRE23_sr_val_inp_subset --eval_gt_path /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt_subset --patch_size 1024 --factor 1.2
    # TLC patch_size 1024 factor 1.2: w/o 24.547   w  23.038
    # CUDA_VISIBLE_DEVICES=6 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name   testing_NAFNet_wTLC-w24Dw960P-23eval25.07-On23eval --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6 --pre_model_dir /root/autodl-tmp/Shadow_Challenge/temp/ --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/Shadow_Challenge/Other/training_NAFNet_0226SC-wRegAug-w23Dw24DwLargeP1440-Arch1-FT-Add23d_Add24test_many/net_epoch_7_PSNR_25.07.pth --global_residual False  --MultiScale False  --evalD 23-eval --eval_in_path  /root/autodl-tmp/ShadowDatasets/NTIRE23_sr_val_inp_subset --eval_gt_path /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt_subset --patch_size 1024 --factor 1.5

    # 250215去反射测试 CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name   SIRR_testing_NAFNet_wTLC_EMA__epoch_599_PSNR_29.57 --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6  --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/ckpt/SIRR_training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4/net_EMA__epoch_599_PSNR_29.57.pth --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/val_100/blended --eval_gt_path /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/val_100/blended --patch_size 256 --factor 1.5 --unified_path /root/autodl-tmp/result/

    # SIRR_training_NAFNet_0219-wRegAug-w25DwP256-lr1e-4-finetune-TLC
# CUDA_VISIBLE_DEVICES=0 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name  SIRR_training_NAFNet_0219-wRegAug-w25DwP256-lr1e-4-finetune-TLC --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6  --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/ckpt/SIRR_training_NAFNet_0218-P256-lr1e-4_PSNR_29.57.pth_finetune/net_EMA__epoch_1183_PSNR_33.53.pth --global_residual False  --MultiScale False  --evalD 25-test --eval_in_path /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/val_100/blended --eval_gt_path /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/val_100/blended --unified_path /root/autodl-tmp/result/ --patch_size 256 --factor 1.5 
 
# lighting_training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name   lighting_testing_NAFNet_0213-wRegAug-w25DwP256-lr4e-4_TLC --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6  --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/lighting_training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4/net_EMA__epoch_598_PSNR_24.91.pth  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/lighting/validation_in --eval_gt_path /root/autodl-tmp/data/lighting/validation_in --patch_size 256 --factor 0.8 --unified_path /root/autodl-tmp/result/

# training_NAFNet_0219-wRegAug-w25Dw24Dw23DwP256-lr4e-4net_EMA__epoch_582_PSNR_25.97.pth
# CUDA_VISIBLE_DEVICES=0 python /root/ShadowChallenge/testing_SCwNAFNet_wTLC.py --experiment_name   testing_NAFNet_0219-wRegAug-w23Dw24Dw25DwP256-PSNR_25.97.pth_TLC --model nafnet --enc_blks 1 1 1 28 --dec_blks 1 1  1  1  --base_channel 32  --img_channel 3 --num_res 6  --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/training_NAFNet_0213-wRegAug-w25Dw24Dw23DwP256-lr4e-4/net_EMA__epoch_582_PSNR_25.97.pth  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --patch_size 256 --factor 0.8 --unified_path /root/autodl-tmp/result/

# IFBlend_ambient6k_best_checkpoint.pt
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_SCwIFBlend.py --experiment_name   IFBlend_ambient6k_best_checkpoint__pt --model IFBlend --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp_100x75 --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp_100x75 --unified_path /root/autodl-tmp/result/

# lighting_infer_IFBlend_ambient6k_best_checkpoint_pt_0220
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_ALNwIFBlend.py --experiment_name   lighting_infer_IFBlend_ambient6k_best_checkpoint_pt_0220 --model IFBlend --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/lighting/validation_in --eval_gt_path /root/autodl-tmp/data/lighting/validation_in --unified_path /root/autodl-tmp/result/

# SC_infer_IFBlend_ambient6k_best_checkpoint_pt_0221  直接resize
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_SCwIFBlend.py --experiment_name   SC_infer_IFBlend_ambient6k_best_checkpoint_pt_0221 --model IFBlend --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --unified_path /root/autodl-tmp/result/

# SC_infer_IFBlend_ambient6k_best_checkpoint_pt_0222  网络结构内padding
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_SCwIFBlend_even.py --experiment_name   SC_infer_IFBlend_ambient6k_best_checkpoint_pt_0222 --model IFBlend --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --unified_path /root/autodl-tmp/result/

# test_SC_finetune_IFBlend_para_0223-wRegAug-w25DwP256-lr2e-4-L1+0.7ssim  网络结构内padding
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/testing_SCwIFBlend_even.py --experiment_name   test_SC_finetune_IFBlend_para_0223-wRegAug-w25DwP256-lr2e-4-L1+0.7ssim --model IFBlend --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/SC_finetune_IFBlend_para_0223-wRegAug-w25DwP256-lr2e-4-L1+0.7ssim/net_EMA_epoch_512.pth  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --unified_path /root/autodl-tmp/result/

# ------------------------------------0312---------------------------------------
# test_SC-ft-IFBlend-0312-w25242324val23val-bs2wP512-lr0.0002-L1+0.7ssim  网络结构内padding
# CUDA_VISIBLE_DEVICES=0 python /root/ShadowChallenge/testing_SCwIFBlend_even.py --experiment_name   test_SC-ft-IFBlend-0312-w25242324val23val-bs2wP512-lr0.0002-L1+0.7ssim-ep149 --model IFBlend --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/SC-ft-IFBlend-0312-w25+24+23+24val+23val-bs2wP512-lr0.0002-L1+0.7ssim/net_EMA__epoch_149_PSNR_29.04.pth  --global_residual False  --MultiScale False  --evalD 25-eval --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_valid_inp --unified_path /root/autodl-tmp/result/
# -----------------------------------------------------------------------------------
# -----------------------------------0312 Testing-----------------------------------------
# test_SC-ft-IFBlend-0312-w25242324val23val-bs2wP512-lr0.0002-L1+0.7ssim  网络结构内padding
# CUDA_VISIBLE_DEVICES=0 python /root/ShadowChallenge/testing_SCwIFBlend_even.py --experiment_name   Testing_SC-ft-IFBlend-0312-w25242324val23val-bs2wP512-lr0.0002-L1+0.7ssim-ep449 --model IFBlend --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/ckpt/SC-ft-IFBlend-0312-w25+24+23+24val+23val-bs2wP512-lr0.0002-L1+0.7ssim/net_EMA__epoch_449_PSNR_30.26.pth  --evalD 25-Testing --eval_in_path  /root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp --eval_gt_path /root/autodl-tmp/data/shadow/ntire25_sh_rem_test_inp --unified_path /root/autodl-tmp/result/
# ----------------------------------------------------------------------------------------
#########################################jrx################################
# CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/ShadowChallenge/testing_ALNwhite_mamba_jrx.py --experiment_name   Testing_distill_EVSSM_EMA_0__epoch_499 --model EVSSM --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/ckpt/training_distill_EVSSM_0313-white-L1+0.02fft-lr1e-4_0/net_EMA__epoch_499_PSNR_41.04.pth  --evalD 25-Testing --eval_in_path  /root/autodl-tmp/Dataset/white/ntire26_aln_test_in --eval_gt_path /root/autodl-tmp/Dataset/white/ntire26_aln_test_in --unified_path /root/autodl-tmp/result/

    if args.model == 'IFBlend':
        from networks.Ifblend_arch_even import IFBlend
        net = IFBlend(16, use_gcb=True).to(device)

    elif args.model == 'EVSSM':
        print(111)
        from networks.EVSSM_arch import EVSSM
        
        if args.enc_blks and len(args.enc_blks) == 3:
            current_num_blocks = args.enc_blks
        else:
            current_num_blocks = [4, 6, 8] # 推荐的默认配置，显存够大可以用 [6, 6, 12]
            print(f"⚠️ Warning: args.enc_blks {args.enc_blks} not match EVSSM requirement (3 stages). Using default: {current_num_blocks}")
            logging.info(f"Using default num_blocks: {current_num_blocks}")
        
        # 这里的参数完全根据你预训练模型 .pth 里的张量尺寸反推出来
        net = EVSSM(
        inp_channels=args.img_channel,      # 通常是 3
        out_channels=3,                     # 输出 RGB
        dim=args.base_channel,              # 基础通道数，建议 32 或 48
        num_blocks=current_num_blocks,      # 每一层的块数
        ffn_expansion_factor=2.66,          # 这是一个经验值，Mamba 类网络常用 2.66 或 3
        bias=False                          # 是否使用偏置
        ).to(device)

    
    if args.models_ensemble:
        net = merge_models(net=net, folder_path=args.pre_model_dir)
        print('-----'*20,'successfully load merged weights!!!!! (with models_ensemble)')

    else:
        net.load_state_dict(torch.load(args.pre_model), strict=True)
        print('-----'*20,'successfully load pre-trained weights!!!!! (without models_ensemble)')
        # load_model_checkpoint = args.pre_model : .pth/.pt
        # IFBlend_checkpoint = torch.load(args.pre_model)
        # net.load_state_dict(IFBlend_checkpoint['model_state_dict'], strict=True)
        # print('-----'*20,'successfully load pre-trained weights!!!!! (without models_ensemble)')

    # # 模型训练时使用 DataParallel 包装导致参数名前缀增加 module.
    # IFBlend_checkpoint = torch.load(args.pre_model)
    # IFBlend_state_dict = IFBlend_checkpoint['model_state_dict']
    # # 移除参数名的 'module.' 前缀（适配 DataParallel 训练保存的模型）
    # new_IFBlend_state_dict = {}
    # for k, v in IFBlend_state_dict.items():
    #     if k.startswith('module.'):
    #         new_k = k[7:]
    #         new_IFBlend_state_dict[new_k] = v
    #     else:
    #         new_IFBlend_state_dict[k] = v
    # net.load_state_dict(new_IFBlend_state_dict, strict=True)

            
    #net.load_state_dict(torch.load(args.pre_model), strict=True)
    #print('-----'*8, 'successfully load pre-trained weights!!!!!','-----'*8)
    #net.to(device)
    
    print_param_number(net)
    # ckpt_path = '/root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt'
    # param_keys_folder_path = '/root/autodl-tmp/ckpt/ckpt_state_dict'
    # model_name = 'IFBlend'
    # save_parameters(net, ckpt_path, param_keys_folder_path, model_name)

    eval_loader  = get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path)

    inference(net= net,eval_loader = eval_loader, Dname = args.evalD , save_result = True)
    # test_Ifblend(net= net,eval_loader = eval_loader, Dname = args.evalD , save_result = True)
    with open(results_mertircs, 'a') as file:
        file.write('-=-='*50)
    if args.inputs_ensemble:
        test_wInputEnsemble(net= net,eval_loader = eval_loader, Dname =   args.evalD , save_result = True)