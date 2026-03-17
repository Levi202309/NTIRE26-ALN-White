import time,torchvision,argparse,logging,sys,os,gc
import torch,random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters,print_args_parameters,Lion
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from utils.EMA import EMA

from datasets.datasets_pairs import my_dataset,my_dataset_eval,my_dataset_wTxt,my_dataset_wTxt_whole_png,my_dataset_wTxt_whole_png_H_W
# from networks.NAFNet_arch import NAFNet,NAFNet_wIN

sys.path.append(os.getcwd())
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,default= "training_SC_wNAF") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/root/autodl-tmp/ckpt/') # save path
#parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_path', type=str,default= '/root/autodl-tmp/') # 选择训练数据的位置
parser.add_argument('--training_path_txt', nargs='*', help='a list of strings') # 选择具体训练数据(配对string)
parser.add_argument('--writer_dir', type=str, default= '/root/tf-logs/')# tbloger的位置
# 验证集用24valid
parser.add_argument('--eval_in_path', type=str,default= '/root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/')
parser.add_argument('--eval_gt_path', type=str,default= '/root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/')

#training setting
parser.add_argument('--EPOCH', type=int, default= 150)
parser.add_argument('--T_period', type=int, default= 50)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default= 2)
parser.add_argument('--Crop_patches', default= [1000, 750], nargs='+', type=int, help='List of integers') 
parser.add_argument('--Crop_patches_1', type=int, default= 128)
parser.add_argument('--Crop_patches_2', type=int, default= 128)
parser.add_argument('--learning_rate', type=float, default= 0.0004)
parser.add_argument('--print_frequency', type=int, default= 50)
parser.add_argument('--SAVE_Inter_Results', type=str2bool, default= False)
#during training
parser.add_argument('--max_psnr', type=int, default= 15)
parser.add_argument('--fix_sampleA', type=int, default= 30000)

parser.add_argument('--debug', type=str2bool, default= False)

parser.add_argument('--Aug_regular', type=str2bool, default= False)
#training setting (arch)
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

#loss  
parser.add_argument('--base_loss', type=str, default= 'char')
parser.add_argument('--addition_loss', type=str, default= 'VGG')
parser.add_argument('--addition_loss_coff', type=float, default= 0.2)
parser.add_argument('--weight_coff', type=float, default= 10.0)

# load load_pre_model
parser.add_argument('--load_pre_model', type=str2bool, default= False)
parser.add_argument('--pre_model', type=str, default= '')

#optim
parser.add_argument('--optim', type=str, default= 'adam')

#training 
parser.add_argument('--use_gradient_accumulation', type=str2bool, default= False)  
parser.add_argument('--accumulation_steps', type=int, default= 2)
#ema

parser.add_argument('--ema_decay', type = float, default= 0.999 )
parser.add_argument('--ema', type=str2bool, default= False)



args = parser.parse_args()
# print all args params!!!
print_args_parameters(args)


if args.debug ==True:
    fix_sampleA = 400
else:
    fix_sampleA = args.fix_sampleA


exper_name =args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    #os.mkdir(args.writer_dir)
    os.makedirs(args.writer_dir, exist_ok=True)
    
unified_path = args.unified_path
SAVE_PATH =unified_path  + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH,exist_ok=True)
if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = SAVE_PATH +'Inter_Temp_results/'
    if not os.path.exists(SAVE_Inter_Results_PATH):
        #os.mkdir(SAVE_Inter_Results_PATH)
        os.makedirs(SAVE_Inter_Results_PATH,exist_ok=True)

logging.basicConfig(filename=SAVE_PATH + args.experiment_name + '.log', level=logging.INFO)

logging.info('======================'*2 + 'args: parameters'+'======================'*2 )
for k in args.__dict__:
    logging.info(k + ": " + str(args.__dict__[k]))
logging.info('======================'*2 + 'args: parameters'+'======================'*2 )
    

trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

logging.info(f'begin training! ')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("=="*50)#Fore.RED +
print( "Check val-RD pairs ???:",os.listdir(args.eval_in_path)==os.listdir(args.eval_gt_path),
      '//      len eval:',len(os.listdir(args.eval_gt_path)))
print("=="*50)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test(net,eval_loader,epoch =1,max_psnr_val=26 ,Dname = 'S'):
    net.eval()
    with torch.no_grad():

        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()

        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            # print(f"✅ text输入尺寸: {inputs.shape}")
            labels = Variable(label).to(device)
            outputs = net(inputs)
            Avg_Meters_evaling.update({ 'eval_output_psnr': compute_psnr(outputs, labels),
                                        'eval_input_psnr': compute_psnr(inputs, labels) })
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        writer.add_scalars(exper_name + '/testing', {'eval_PSNR_Output': Final_output_PSNR,
                                                     'eval_PSNR_Input': Final_input_PSNR }, epoch)
        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
            # saving pre-weighted
            torch.save(net.state_dict(),
                       SAVE_PATH + f'net_epoch_{epoch}_PSNR_{round(max_psnr_val, 2)}.pth')

        print("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() -st ))
        logging.info("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() -st ))
        
    return max_psnr_val

def save_model(net, epoch=1):
    if epoch in {0, 15, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500}:
        save_path = SAVE_PATH + f'net_epoch_{epoch}_.pth'
        torch.save(net.state_dict(), save_path)
        # 添加打印信息
        print(f"✅ 普通模型已保存至: {save_path} (Epoch {epoch})")


def test_wEMA(net,net_ema, eval_loader, epoch=1, max_psnr_val=26, Dname='S'):
    net.eval()
    net_ema.apply_shadow()

    with torch.no_grad():

        # eval_results =
        Avg_Meters_evaling = AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):  # enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            outputs = net(inputs)
            Avg_Meters_evaling.update({'eval_output_psnr': compute_psnr(outputs, labels),
                                       'eval_input_psnr': compute_psnr(inputs, labels)})
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr']  # / len(eval_loader)
        writer.add_scalars(exper_name + '/testing(EMA) ', {'eval_PSNR_Output': Final_output_PSNR,
                                                     'eval_PSNR_Input': Final_input_PSNR}, epoch)
        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
            # saving pre-weighted
            torch.save(net.state_dict(),
                       SAVE_PATH + f'net_EMA__epoch_{epoch}_PSNR_{round(max_psnr_val, 2)}.pth')

        print(
            "(EMA) epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(
                epoch, Dname, len(eval_loader), round(Final_input_PSNR, 2),
                round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() - st))
        logging.info(
            "(EMA) epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(
                epoch, Dname, len(eval_loader), round(Final_input_PSNR, 2),
                round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() - st))
    net_ema.restore()
    return max_psnr_val

def save_model_wEMA(net, net_ema, epoch):
    if epoch in {0, 15, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500}:
        net_ema.apply_shadow()  # 应用EMA参数
        save_path = SAVE_PATH + f'net_EMA_epoch_{epoch}.pth'
        torch.save(net.state_dict(), save_path)
        # 添加打印信息
        print(f"🔥 EMA模型已保存至: {save_path} (Epoch {epoch})")
        net_ema.restore()  # 恢复原始参数

def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)

def get_training_data( Crop_patches=args.Crop_patches):
    rootA = args.training_path
    rootA_txt1_list = args.training_path_txt
    train_Pre_dataset_list = []
    for idx_dataset in range(len(rootA_txt1_list)):
        # if args.Crop_patches == 1920:
        #     train_Pre_dataset = my_dataset_wTxt_whole_png(rootA, rootA_txt1_list[idx_dataset],
        #                                         crop_size=Crop_patches,
        #                                         fix_sample_A=fix_sampleA,
        #                                         regular_aug=args.Aug_regular)  # threshold_size =  args.threshold_size
        # else:            
        #     train_Pre_dataset = my_dataset_wTxt(rootA, rootA_txt1_list[idx_dataset],
        #                                         crop_size=Crop_patches,
        #                                         fix_sample_A=fix_sampleA,
        #                                         regular_aug=args.Aug_regular)  # threshold_size =  args.threshold_size
        train_Pre_dataset = my_dataset_wTxt_whole_png_H_W(rootA, rootA_txt1_list[idx_dataset],
                                            crop_size=Crop_patches,
                                            fix_sample_A=fix_sampleA,
                                            regular_aug=args.Aug_regular)  # threshold_size =  args.threshold_size
        train_Pre_dataset_list.append(train_Pre_dataset)
    train_pre_datasets = ConcatDataset(train_Pre_dataset_list)
    
    train_loader = DataLoader(dataset=train_pre_datasets, batch_size=args.BATCH_SIZE, num_workers= 8 ,shuffle=True)
    print('len(train_loader):' ,len(train_loader))
    logging.info(f"len(train_loader): {len(train_loader)}")
    return train_loader
    

def get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers= 4)
    return eval_loader
def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))
    logging.info('#generator parameters: %d' % sum(param.numel() for param in net.parameters()))

if __name__ == '__main__':    

    # Finetuning_p960 with only_24test_style_lr4e-5 加大patch到960 加入ntire24_test_fake_naf，ntire24_test_fake_vit，ntire24_test_fake318.txt  换了24.69的预训练权重
    # CUDA_VISIBLE_DEVICES=2 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name training_NAFNet_0226SC-wRegAug-wLargeP960-Arch1-FT-lr4e-5-Add24test_many_only --unified_path /root/autodl-tmp/Shadow_Challenge/  --training_path /root/autodl-tmp/  --training_path_txt   /root/autodl-tmp/ShadowDatasets/ntire24_test_fake_vit.txt /root/autodl-tmp/ShadowDatasets/ntire24_test_fake_naf.txt /root/autodl-tmp/ShadowDatasets/ntire24_test_fake318.txt  --EPOCH 200  --T_period 40 --BATCH_SIZE 1 --Crop_patches 960 --learning_rate 0.00004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6 --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay 0.9995 --load_pre_model True --pre_model /root/autodl-tmp/Shadow_Challenge/training_NAFNet_0226SC-wRegAug-w23Dw24DwLargeP960-Arch1-FT-Add23d_Add24test_many/net\(EMA\)_epoch_0_PSNR_24.69.pth --eval_in_path /root/autodl-tmp/ShadowDatasets/NTIRE23_sr_val_inp_subset/   --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt_subset/  --print_frequency 50 --use_gradient_accumulation True  --accumulation_steps 2



# shadow-250213:初步尝试nafnet，只用25D   PSNR:24.66
 # CUDA_VISIBLE_DEVICES=0 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 22 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6 --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/

 # shadow-初步尝试nafnet，用25D+24D        PSNR:25.04
 # CUDA_VISIBLE_DEVICES=1 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name training_NAFNet_0213-wRegAug-w25Dw24DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt /root/autodl-tmp/data/shadow/ntire2024.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 22 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6 --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/

#  shadow-初步尝试nafnet，用25D+24D+23D        PSNR:25.
 # CUDA_VISIBLE_DEVICES=0 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name training_NAFNet_0213-wRegAug-w25Dw24Dw23DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt /root/autodl-tmp/data/shadow/ntire2024.txt /root/autodl-tmp/data/shadow/ntire2023.txt /root/autodl-tmp/data/shadow/ntire2023_val.txt   --EPOCH 600  --T_period 50 --BATCH_SIZE 22 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6 --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/

# lighting-250213:初步尝试nafnet 5000 images
 # CUDA_VISIBLE_DEVICES=1 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name lighting_training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/lighting/ntire25.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 22 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6  --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/lighting/in_subset/   --eval_gt_path  /root/autodl-tmp/data/lighting/gt_subset/

 # SIRR-250213:初步尝试nafnet 800 images
 # CUDA_VISIBLE_DEVICES=2 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name SIRR_training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/ntire_SIRR25.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 22 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6 --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/train_800/blended_subset/   --eval_gt_path  /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/train_800/transmission_layer_subset/

 # SIRR-250218: 微调 nafnet 800 images
 # CUDA_VISIBLE_DEVICES=1 python /root/ShadowChallenge/training_SC_wNAFNet_wEMA.py --experiment_name SIRR_training_NAFNet_0218-P256-lr1e-4_PSNR_29.57.pth_finetune  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/ntire_SIRR25.txt    --EPOCH 2000  --T_period 50 --BATCH_SIZE 22 --Crop_patches 256 --learning_rate 0.0001 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --enc_blks 1 1 1 28  --dec_blks 1 1  1  1 --base_channel 32 --img_channel 3 --num_res 6 --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/train_800/blended_subset/   --eval_gt_path  /root/autodl-tmp/data/NTIRE2025_Challenge_SIRR/train_800/transmission_layer_subset/ --load_pre_model True --pre_model /root/autodl-tmp/ckpt/SIRR_training_NAFNet_0213-wRegAug-w25DwP256-lr4e-4/net_EMA__epoch_599_PSNR_29.57.pth

# shadow-250218:初步尝试IFBlend，只用25D   PSNR
 # CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name training_IFBlend_0218-wRegAug-w25DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 14 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/ 

# shadow-250219:初步尝试IFBlend，只用25D   PSNR
 # CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name training_IFBlend_0219-wRegAug-w25DwP256-lr4e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 14 --Crop_patches 256 --learning_rate 0.0004 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model False --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/ 

# shadow-250222:IFBlend权重训练参数，只用25D   PSNR
 # CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name finetune_IFBlend_para_0222-wRegAug-w25DwP256-lr2e-4-L1+0.7ssim  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 8 --Crop_patches 256 --learning_rate 0.0002 --addition_loss ssim --addition_loss_coff 0.7 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/  --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt

 # ALN-250222:finetune-IFBlend，用25D /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt
 # CUDA_VISIBLE_DEVICES=2 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name ALN_finetune_IFBlend_para_0222-wRegAug-w25DwP512-lr2e-4-L1+0.7ssime  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/lighting/ntire25.txt  --EPOCH 2000  --T_period 50 --BATCH_SIZE 2 --Crop_patches 512 --learning_rate 0.00025 --addition_loss ssim --addition_loss_coff 0.72 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/lighting/in_subset/   --eval_gt_path   /root/autodl-tmp/data/lighting/gt_subset/ --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt

# SC-250223:finetune-IFBlend，只用25D   PSNR
 # CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name SC_finetune_IFBlend_para_0223-wRegAug-w25DwP256-lr2e-4-L1+0.7ssim  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt    --EPOCH 600  --T_period 50 --BATCH_SIZE 8 --Crop_patches 256 --learning_rate 0.0002 --addition_loss ssim --addition_loss_coff 0.7 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/  --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt

# ALN-250223:finetune-IFBlend，用25D /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt
 # CUDA_VISIBLE_DEVICES=2 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name ALN_finetune_IFBlend_para_0223-wRegAug-w25DwP512-lr2e-4-L1+0.7ssime  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/lighting/ntire25.txt  --EPOCH 2000  --T_period 50 --BATCH_SIZE 2 --Crop_patches 512 --learning_rate 0.00025 --addition_loss ssim --addition_loss_coff 0.72 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/lighting/in_subset/   --eval_gt_path   /root/autodl-tmp/data/lighting/gt_subset/ --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt

# ALN-250223-2:finetune-IFBlend，用25D /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt
 # CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/training_SC_wIFBlend_wEMA.py --experiment_name ALN_finetune_IFBlend_para_0223-2-wRegAug-w25DwP512-lr2e-4-L1+0.7ssime  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/lighting/ntire25.txt  --EPOCH 2000  --T_period 50 --BATCH_SIZE 2 --Crop_patches 512 --learning_rate 0.00005 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss char  --weight_coff 2.0  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/lighting/in_subset/   --eval_gt_path   /root/autodl-tmp/data/lighting/gt_subset/ --pre_model /root/autodl-tmp/ckpt/IFBlend_ambient6k_best_checkpoint.pt

# --optim adamw  L1 + 0.2 fft  lr=2e-4
#  shadow-初步尝试AdaIR，用25D+24D+23D        PSNR:
# CUDA_VISIBLE_DEVICES=3 python /root/ShadowChallenge/training_SC_wAdaIR_wEMA.py --experiment_name training_AdaIR_0305-w25+24+23wP128bs2-L1+0.02fft-lr2e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path /root/autodl-tmp/  --training_path_txt  /root/autodl-tmp/data/shadow/ntire2025.txt /root/autodl-tmp/data/shadow/ntire2024.txt /root/autodl-tmp/data/shadow/ntire2023.txt /root/autodl-tmp/data/shadow/ntire2023_val.txt   --EPOCH 600  --T_period 50 --BATCH_SIZE 8 --Crop_patches 256 --optim adamw --learning_rate 0.0002 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model False --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/data/shadow/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/data/shadow/ntire24_shrem_valid_gt/

#shadow260204-0:28-尝试EVSSM_mamba,用all data
#CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/ShadowChallenge/training_shadow_manba_wEMA_zza.py --experiment_name training_EVSSM_0203-wall_wP256bs8-L1+0.02fft-lr2e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path_txt /root/autodl-tmp/Dataset/Shadow/ntire26_shadow_removal_train.txt  /root/autodl-tmp/Dataset/Shadow/ntire23_sr_train.txt  /root/autodl-tmp/Dataset/Shadow/ntire23_sr_val.txt /root/autodl-tmp/Dataset/Shadow/ntire24_sr_train.txt /root/autodl-tmp/Dataset/Shadow/ntire25_sr_train.txt  --EPOCH 600  --T_period 50 --BATCH_SIZE 4 --Crop_patches 256 --optim adamw --learning_rate 0.0002 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model False --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/Dataset/Shadow/shadow_old/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/Dataset/Shadow/shadow_old/ntire24_shrem_valid_gt/
####################################################################################################################################
# **new   256*14**
# CUDA_VISIBLE_DEVICES=3 python /root/autodl-tmp/ShadowChallenge/training_white_manba_wEMA_zza.py --experiment_name training_EVSSM_0304-whiteall_wP256bs14-L1+0.02fft-lr2e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path_txt /root/autodl-tmp/Dataset/white/26_white_train.txt  --EPOCH 500  --T_period 50 --BATCH_SIZE 14 --Crop_patches 256 --optim adamw --learning_rate 0.0002 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model False --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/Dataset/white/AMBIENT6K/test/RGB/in/   --eval_gt_path  /root/autodl-tmp/Dataset/white/AMBIENT6K/test/RGB/gt/
# test

# CUDA_VISIBLE_DEVICES=3 python /root/autodl-tmp/ShadowChallenge/testing_ALNwhite_mamba_zza.py --experiment_name   Testing_white_mamba_zza_256_14 --model EVSSM --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/ckpt/training_EVSSM_0304-whiteall_wP256bs14-L1+0.02fft-lr2e-4/net_epoch_107_PSNR_19.71.pth  --evalD 26-Testing --eval_in_path  /root/autodl-tmp/Dataset/white/ntire26_aln_valid_in --eval_gt_path /root/autodl-tmp/Dataset/white/ntire26_aln_valid_in --unified_path /root/autodl-tmp/result/

# score:19.36

# train 384*6 2026.03.06 23:10

# CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/ShadowChallenge/training_white_manba_wEMA_zza.py --experiment_name training_EVSSM_0306-whiteall_wP384bs6-L1+0.02fft-lr2e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path_txt /root/autodl-tmp/Dataset/white/26_white_train.txt  --EPOCH 500  --T_period 50 --BATCH_SIZE 6 --Crop_patches 384 --optim adamw --learning_rate 0.0002 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/Dataset/white/AMBIENT6K/test/RGB/in/   --eval_gt_path  /root/autodl-tmp/Dataset/white/AMBIENT6K/test/RGB/gt/  --pre_model  /root/autodl-tmp/ckpt/training_EVSSM_0306-whiteall_wP384bs6-L1+0.02fft-lr2e-4/net_epoch_231_PSNR_20.61.pth



# CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/ShadowChallenge/training_white_manba_wEMA_zza.py --experiment_name training_EVSSM_0306-whiteall_wP384bs6-L1+0.02fft-lr2e-4  --unified_path /root/autodl-tmp/ckpt/  --training_path_txt /root/autodl-tmp/Dataset/white/26_white_train.txt  --EPOCH 500  --T_period 50 --BATCH_SIZE 6 --Crop_patches 384 --optim adamw --learning_rate 0.0002 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/Dataset/white/AMBIENT6K/test/RGB/in/   --eval_gt_path  /root/autodl-tmp/Dataset/white/AMBIENT6K/test/RGB/gt/  --pre_model  /root/autodl-tmp/ckpt/training_EVSSM_0306-whiteall_wP384bs6-L1+0.02fft-lr2e-4/net_epoch_231_PSNR_20.61.pth

    ####################################jrx############################################
# CUDA_VISIBLE_DEVICES=5 python /root/autodl-tmp/ShadowChallenge/training_white_mamba_wEMA_jrx.py --experiment_name training_distill_EVSSM_0313-white-L1+0.02fft-lr1e-4_0  --unified_path /root/autodl-tmp/ckpt/  --training_path_txt /root/autodl-tmp/Dataset/white/26_white_test_pseudo_NAFNetBest.txt --EPOCH 500  --T_period 50 --BATCH_SIZE 1 --Crop_patches 1024 768 --optim adamw --learning_rate 0.0001 --addition_loss fft --addition_loss_coff 0.02 --SAVE_Inter_Results False  --Aug_regular True  --load_pre_model True --base_loss L1  --global_residual False  --MultiScale False --ema True  --ema_decay   0.9995   --eval_in_path /root/autodl-tmp/Dataset/white/ntire26_aln_test_in   --eval_gt_path  /root/autodl-tmp/Dataset/white/ntire26white_nafnet_best_test_pseudo/w.oInputEnsemble  --pre_model  /root/autodl-tmp/ckpt/training_EVSSM_0306-whiteall_wP384bs6-L1+0.02fft-lr2e-4/net_epoch_231_PSNR_20.61.pth


####################################################################################################################################
    # from networks.NAFNet_arch import NAFNet
    # net = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res, enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,global_residual =  args.global_residual, MultiScale = args.MultiScale,drop_flag = args.drop_flag,  drop_rate = args.drop_rate, kernel_size =args.kernel_size )
    # from networks.Ifblend_arch import IFBlend
    # net = IFBlend(16, use_gcb=True)
    # from networks.Ifblend_arch_even import IFBlend
    # net = IFBlend(16, use_gcb=True).to(device)
    from networks.EVSSM_arch import EVSSM

    # 2. 准备 num_blocks 参数
    # EVSSM 需要一个包含3个整数的列表，例如 [4, 6, 8] 代表三层 Encoder 的深度。
    # 你的 args.enc_blks 可能是 NAFNet 的格式 (如 [1, 1, 1, 28])，这不匹配。
    # 逻辑：如果命令行传了3个参数，就用命令行的；否则使用默认配置。
    if args.enc_blks and len(args.enc_blks) == 3:
        current_num_blocks = args.enc_blks
    else:
        current_num_blocks = [4, 6, 8] # 推荐的默认配置，显存够大可以用 [6, 6, 12]
        print(f"⚠️ Warning: args.enc_blks {args.enc_blks} not match EVSSM requirement (3 stages). Using default: {current_num_blocks}")
        logging.info(f"Using default num_blocks: {current_num_blocks}")

    # 3. 实例化模型
    net = EVSSM(
        inp_channels=args.img_channel,      # 通常是 3
        out_channels=3,                     # 输出 RGB
        dim=args.base_channel,              # 基础通道数，建议 32 或 48
        num_blocks=current_num_blocks,      # 每一层的块数
        ffn_expansion_factor=2.66,          # 这是一个经验值，Mamba 类网络常用 2.66 或 3
        bias=False                          # 是否使用偏置
    )
    
    start_epoch = 0 

    if args.load_pre_model:
        # net.load_state_dict(torch.load(args.pre_model,map_location='cuda:0'), strict=True)
        # print('-----'*20,'successfully load pre-trained weights!!!!!')
        # logging.info('-----'*20,'successfully load pre-trained weights!!!!!')
        
        # 模型训练时使用 DataParallel 包装导致参数名前缀增加 module.
        IFBlend_checkpoint = torch.load(args.pre_model,map_location='cuda:0')
        #IFBlend_state_dict = IFBlend_checkpoint['model_state_dict']

        if isinstance(IFBlend_checkpoint, dict) and 'model_state_dict' in IFBlend_checkpoint:
            # 情况 A: 这是一个包含元数据的 Checkpoint
            state_dict = IFBlend_checkpoint['model_state_dict']
            print("Detected Checkpoint format (with 'model_state_dict').")
        else:
            # 情况 B: 这是一个纯参数字典 (你现在的 net_epoch_200_.pth 属于这种情况)
            state_dict = IFBlend_checkpoint
            print("Detected pure State Dict format.")

        # 移除参数名的 'module.' 前缀（适配 DataParallel 训练保存的模型）
        new_IFBlend_state_dict = {}
        #for k, v in IFBlend_state_dict.items():
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_k = k[7:]
                new_IFBlend_state_dict[new_k] = v
            else:
                new_IFBlend_state_dict[k] = v 
        net.load_state_dict(new_IFBlend_state_dict, strict=True)
        print('-----'*20,'successfully load pre-trained weights!!!!!')
        logging.info('-----'*20 + 'successfully load pre-trained weights!!!!!')

        # try:
        #     # 分割字符串找到 'epoch' 后面的数字
        #     file_name = os.path.basename(args.pre_model)
        #     # 简单粗暴的提取方法，根据你的文件名格式调整
        #     start_epoch = int(file_name.split('epoch_')[1].split('_')[0]) 
            
        #     # 如果你想接着上一轮跑，通常加 1
        #     start_epoch += 1
        #     print(f"🔄 恢复训练进度，将从 Epoch {start_epoch} 开始！")
        # except Exception as e:
        #     print(f"⚠️ 无法从文件名提取 Epoch，将从 0 开始。错误: {e}")
    
    '''
    if torch.__version__[0] == '2':
        net = torch.compile(net)
        print('-----'*20, torch.__version__)
        logging.info('-----'*20, torch.__version__)
    '''
    net.to(device)
    print_param_number(net)
    
    if args.ema:
        net_ema = EMA(model=net, decay=args.ema_decay)
        net_ema.register()
        
    train_loader = get_training_data()
    eval_loader  = get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path)

    
    if args.optim.lower() == 'adamw':
        optimizerG = optim.AdamW(net.parameters(), lr=args.learning_rate,betas=(0.9,0.999))
    elif args.optim.lower() == 'lion':
        optimizerG = Lion(net.parameters(), lr=args.learning_rate,betas=(0.9,0.999))
    else:
        # optimizerG = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9,0.999) )
        # IFBlend 权重
        optimizerG = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.5,0.999) )

    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period,  T_mult=1) #ExponentialLR(optimizerG, gamma=0.98)


    if args.base_loss.lower() == 'char':
        base_loss = losses.CharbonnierLoss()
    elif args.base_loss.lower() == 'weightedchar':
        base_loss = losses.WeightedCharbonnierLoss(eps=1e-4, weight = args.weight_coff)
    else:
        base_loss = nn.L1Loss()

    if args.addition_loss.lower()  == 'vgg':
        criterion = losses.VGGLoss()
    elif args.addition_loss.lower()  == 'fft':
        criterion = losses.fftLoss()
    elif args.addition_loss.lower()  == 'ssim':
        criterion = losses.SSIMLoss()   
        
    criterion_depth = nn.L1Loss()

    # recording values! ( training process~)
    running_results = { 'iter_nums' : 0  , 'max_psnr_valD': 0 , 'max_psnr_valD(ema)': 0 }
    # 'max_psnr_valD' :  args.max_psnr, 'total_loss': 0.0,  'total_loss1': 0.0,
    #                         'total_loss2': 0.0,  'input_PSNR_all': 0.0,  'train_PSNR_all': 0.0,

    Avg_Meters_training = AverageMeters()

    #iter_nums = 0
    for epoch in range(start_epoch, args.EPOCH):
        scheduler.step(epoch)
        st = time.time()

        for i,train_data in enumerate(train_loader,0):
            data_in, label, img_name = train_data
            if i ==0:
                print(f" input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" input.size: {data_in.size()}, gt.size: {label.size()}")
            running_results['iter_nums'] +=1

            net.train()
            net.zero_grad()
            optimizerG.zero_grad()

            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            train_output = net(inputs)
            
            # calcuate metrics
            input_PSNR = compute_psnr(inputs, labels)
            trian_PSNR = compute_psnr(train_output, labels)

            loss1 =  base_loss(train_output, labels) # losses.multi_scale_losses(train_output, labels, base_loss )#  ()

            if args.addition_loss.lower() == 'vgg':
                loss2 =  args.addition_loss_coff * criterion(train_output, labels)  # losses.multi_scale_losses(train_output, labels, criterion )
                g_loss = loss1  + loss2
                loss3 = loss1
            elif args.addition_loss.lower() == 'fft':
                loss2 =  args.addition_loss_coff * criterion(train_output, labels)
                g_loss = loss1  + loss2
                loss3 = loss1
            elif args.addition_loss.lower() == 'ssim':
                loss2 =  args.addition_loss_coff * criterion(train_output, labels)
                g_loss = loss1  + loss2
                loss3 = loss1
            else:
                g_loss = loss1 #+ loss2
                loss2 = loss1   # 0.1 * criterion(train_output, labels)
                loss3 = loss1

            # if args.depth_loss :
            #     loss3 = args.lam_DepthLoss * criterion_depth(train_output, labels)
            #     g_loss = loss1 + loss2 + loss3
            # else:
            #     g_loss = loss1 + loss2
            #     loss3 = loss1

            Avg_Meters_training.update({'total_loss': g_loss.item(),  'total_loss1': loss1.item(),   'total_loss2': loss2.item(),
                                        'total_loss3': loss3.item(), 'input_PSNR_all': input_PSNR, 'train_PSNR_all': trian_PSNR
                                         })
            g_loss.backward()
            #optimizerG.step()
            
            if args.use_gradient_accumulation: # --use_gradient_accumulation True  --accumulation_steps 2 
                # Accumulate gradients and update weights every accumulation_steps
                if (i + 1) % args.accumulation_steps == 0:
                    optimizerG.step()
                    net.zero_grad()
                    optimizerG.zero_grad()
            else:
                # Update weights after every batch
                optimizerG.step()
                #net.zero_grad()
                #optimizerG.zero_grad()
            
            if args.ema:
                net_ema.update()

            if (i+1) % args.print_frequency ==0 and i >1:
                writer.add_scalars(exper_name +'/training' ,{'PSNR_Output':  Avg_Meters_training['train_PSNR_all'], 'PSNR_Input':  Avg_Meters_training['input_PSNR_all'], } , running_results['iter_nums'])
                writer.add_scalars(exper_name +'/training' ,{'total_loss': Avg_Meters_training['total_loss']  ,'loss1_char':  Avg_Meters_training['total_loss1'] , 'loss2': Avg_Meters_training['total_loss2'],
                                                             'loss3': Avg_Meters_training['total_loss3']  } , running_results['iter_nums'])
                print(
                    "epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f, avg_loss:%.5f],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f" %
                    (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), loss1.item(),
                     loss2.item(), loss3.item(), Avg_Meters_training['total_loss'], input_PSNR, trian_PSNR, time.time() - st))
                logging.info(
                    "epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f, avg_loss:%.5f],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f" %
                    (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), loss1.item(),
                     loss2.item(), loss3.item(), Avg_Meters_training['total_loss'], input_PSNR, trian_PSNR, time.time() - st))
                
                st = time.time()
                if args.SAVE_Inter_Results:
                    save_path = SAVE_Inter_Results_PATH + str(running_results['iter_nums']) + '.jpg'
                    save_imgs_for_visual(save_path, inputs, labels, train_output)

        # evaluation
        running_results['max_psnr_valD'] = test(net= net,eval_loader = eval_loader,epoch=epoch,max_psnr_val = running_results['max_psnr_valD'], Dname = 'evalD')
        if args.ema:
            running_results['max_psnr_valD(ema)'] = test_wEMA(net= net,net_ema=net_ema, eval_loader = eval_loader,epoch=epoch,max_psnr_val = running_results['max_psnr_valD(ema)'], Dname = 'evalD(ema)')
        
        # 仅保存模型
        save_model(net= net,epoch=epoch)
        if args.ema:
            save_model_wEMA(net=net,net_ema=net_ema,epoch=epoch)