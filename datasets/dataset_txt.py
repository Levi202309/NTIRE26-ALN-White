import torch,os,random
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader


def read_txt(txt_name = 'ISTD.txt',sample= True,sample_num=800000):
    path_in = []
    path_gt = []
    path_mask = []
    paths =[]
    pre_sample = sample_num
    with open(txt_name, 'r') as f:  # RealSnow
        for line in f:
            paths.append(line.strip('\n'))
    if sample:
        if sample_num > len(paths):
            sample_num = len(paths)
        print('Want to Sample:{}----Actually Sample Numbers:{}----'.format(pre_sample, sample_num))
        paths_random = random.sample(paths, sample_num)
    else:
        paths_random = paths
    for path in paths_random:
        path_in.append(path.strip('\n').split(' ')[0])
        path_gt.append(path.strip('\n').split(' ')[1])
        path_mask.append(path.strip('\n').split(' ')[2])

    return path_in,path_gt,path_mask


class my_dataset_threeIn(Dataset):
    def __init__(self,root,root_txt,crop_size =256, Crop = False, factor = 16,sample= True,fix_sample =10000):
        super(my_dataset_threeIn,self).__init__()

        self.sample = sample
        self.fix_sample = fix_sample
        in_files, gt_files, mask_files = read_txt( root_txt, sample = self.sample, sample_num = self.fix_sample )  # os.listdir(rootA_in)
        self.imgs_in = [root + k for k in in_files]  # os.path.join(rootA_in, k)
        self.imgs_gt = [root + k for k in gt_files]
        self.imgs_mask = [root + k for k in mask_files]

        self.crop_size = crop_size
        self.Crop = Crop
        self.factor = factor
    def __getitem__(self, index):
        in_img =  np.array( Image.open(self.imgs_in[index] ) )
        gt_img =  np.array(Image.open(self.imgs_gt[index] ) )
        mask_img =  np.array(Image.open(self.imgs_mask[index] ) )

        #print(self.Crop)
        if self.Crop:
            data_IN, data_GT, data_MASK = self.train_transform_wCrop(in_img, gt_img, mask_img, self.crop_size)
        else:
            data_IN, data_GT, data_MASK = self.train_transform(in_img, gt_img, mask_img)

        _, h, w = data_GT.shape
        if (h % self.factor != 0) or (w % self.factor != 0):
            data_GT = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_GT)
            data_IN = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_IN)
            data_MASK = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_MASK)

        return data_IN, data_GT, data_MASK

    def train_transform_wCrop(self, img, label, mask, patch_size=256):
        ih, iw,_ = img.shape

        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))

        img = img[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]
        mask = mask[iy:iy + patch_size, ix: ix + patch_size]

        # mode = random.randint(0, 7)
        # img = np.expand_dims(img, axis=2)
        # label = np.expand_dims(label, axis=2)
        # img = self.augment_img(img, mode=mode)
        # label = self.augment_img(label, mode=mode)
        # img = img.copy()
        # label = label.copy()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)
        mask = transform(mask)

        return img, label, mask

    def train_transform(self, img, label, mask):

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)
        mask = transform(mask)

        return img, label, mask

    def __len__(self):
        return len(self.imgs_in)


class my_dataset_threeIn_wAug(Dataset):
    def __init__(self, root, root_txt, crop_size=256, Crop=False, factor=16, sample=True, fix_sample=10000, Aug = True):
        super(my_dataset_threeIn_wAug, self).__init__()

        self.sample = sample
        self.fix_sample = fix_sample
        in_files, gt_files, mask_files = read_txt(root_txt, sample=self.sample,
                                                  sample_num=self.fix_sample)  # os.listdir(rootA_in)
        self.imgs_in = [root + k for k in in_files]  # os.path.join(rootA_in, k)
        self.imgs_gt = [root + k for k in gt_files]
        self.imgs_mask = [root + k for k in mask_files]

        self.crop_size = crop_size
        self.Crop = Crop
        self.factor = factor

        self.Aug = Aug
        self.ind_transform = None

        self.joint_transform = transforms.Compose(
                [transforms.ToTensor(),
                 ]
            )

    def __getitem__(self, index):
        in_img = np.array(Image.open(self.imgs_in[index]))
        gt_img = np.array(Image.open(self.imgs_gt[index]))
        mask_img = np.array(Image.open(self.imgs_mask[index]))

        #print('0',in_img.shape, mask_img.shape)

        # print(self.Crop)
        if self.Crop:
            data_IN, data_GT, data_MASK = self.train_transform_wCrop(in_img, gt_img, mask_img, self.crop_size)
        else:
            data_IN, data_GT, data_MASK = self.train_transform(in_img, gt_img, mask_img)

        _, h, w = data_GT.shape
        if (h % self.factor != 0) or (w % self.factor != 0):
            data_GT = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_GT)
            data_IN = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_IN)
            data_MASK = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(
                data_MASK)
        return data_IN, data_GT, data_MASK

    def train_transform_wCrop(self, img, label, mask, patch_size=256):
        ih, iw, _ = img.shape

        #print('1',mask.shape)
        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))

        img = img[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]
        mask = mask[iy:iy + patch_size, ix: ix + patch_size]
        #print('2',mask.shape)

        if self.Aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            mask = self.augment_img(mask, mode=mode)
            img = img.copy()
            label = label.copy()
            mask = mask.copy()


        img = self.joint_transform((Image.fromarray(img)))
        label = self.joint_transform((Image.fromarray(label)))
        mask = self.joint_transform(Image.fromarray(mask))

        return img, label, mask

    def train_transform(self, img, label, mask):
        if self.Aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            mask = self.augment_img(mask, mode=mode)
            img = img.copy()
            label = label.copy()
            mask = mask.copy()


        img = self.joint_transform((Image.fromarray(img)))
        label = self.joint_transform((Image.fromarray(label)))
        mask = self.joint_transform(Image.fromarray(mask))

        return img, label, mask

    def __len__(self):
        return len(self.imgs_in)

    def augment_img(self, img, mode=0):
        """图片随机旋转"""
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.flipud(np.rot90(img, k=3)))


class my_dataset_threeIn_wAug_w256(Dataset):
    def __init__(self, root, root_txt, crop_size=256, Crop=False, factor=16, sample=True, fix_sample=10000, Aug = True):
        super(my_dataset_threeIn_wAug_w256, self).__init__()

        self.sample = sample
        self.fix_sample = fix_sample
        in_files, gt_files, mask_files = read_txt(root_txt, sample=self.sample,
                                                  sample_num=self.fix_sample)  # os.listdir(rootA_in)
        self.imgs_in = [root + k for k in in_files]  # os.path.join(rootA_in, k)
        self.imgs_gt = [root + k for k in gt_files]
        self.imgs_mask = [root + k for k in mask_files]

        self.crop_size = crop_size
        self.Crop = Crop
        self.factor = factor

        self.Aug = Aug
        self.ind_transform = None

        self.joint_transform = transforms.Compose(

                [   transforms.Resize([256,256]),
                    transforms.ToTensor(),
                 ]
            )

    def __getitem__(self, index):
        in_img = np.array(Image.open(self.imgs_in[index]))
        gt_img = np.array(Image.open(self.imgs_gt[index]))
        mask_img = np.array(Image.open(self.imgs_mask[index]))

        #print('0',in_img.shape, mask_img.shape)

        # print(self.Crop)
        if self.Crop:
            data_IN, data_GT, data_MASK = self.train_transform_wCrop(in_img, gt_img, mask_img, self.crop_size)
        else:
            data_IN, data_GT, data_MASK = self.train_transform(in_img, gt_img, mask_img)

        _, h, w = data_GT.shape
        if (h % self.factor != 0) or (w % self.factor != 0):
            data_GT = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_GT)
            data_IN = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_IN)
            data_MASK = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(
                data_MASK)
        return data_IN, data_GT, data_MASK

    def train_transform_wCrop(self, img, label, mask, patch_size=256):
        ih, iw, _ = img.shape

        #print('1',mask.shape)
        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))

        img = img[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]
        mask = mask[iy:iy + patch_size, ix: ix + patch_size]
        #print('2',mask.shape)

        if self.Aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            mask = self.augment_img(mask, mode=mode)
            img = img.copy()
            label = label.copy()
            mask = mask.copy()


        img = self.joint_transform((Image.fromarray(img)))
        label = self.joint_transform((Image.fromarray(label)))
        mask = self.joint_transform(Image.fromarray(mask))

        return img, label, mask

    def train_transform(self, img, label, mask):
        if self.Aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            mask = self.augment_img(mask, mode=mode)
            img = img.copy()
            label = label.copy()
            mask = mask.copy()


        img = self.joint_transform((Image.fromarray(img)))
        label = self.joint_transform((Image.fromarray(label)))
        mask = self.joint_transform(Image.fromarray(mask))

        return img, label, mask

    def __len__(self):
        return len(self.imgs_in)

    def augment_img(self, img, mode=0):
        """图片随机旋转"""
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.flipud(np.rot90(img, k=3)))

class my_dataset_threeIn_test(Dataset):
    def __init__(self,root_in,root_label,root_mask,transform =None, factor = 16):
        super(my_dataset_threeIn_test,self).__init__()
        #in_imgs
        in_files = sorted(os.listdir(root_in))
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        gt_files = sorted(os.listdir(root_label))
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        #mask_imgs
        mask_files = sorted(os.listdir(root_mask))
        self.imgs_mask = [os.path.join(root_mask, k) for k in mask_files]

        print('check dataset: (in_files == gt_files)',in_files == gt_files )
        print('check dataset: (in_files == mask_files)',in_files == mask_files )

        self.transform = transform
        self.factor = factor

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]
        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)
        mask_img_path = self.imgs_mask[index]
        mask_img = Image.open(mask_img_path)
        if self.transform:
            data_IN = self.transform(in_img)
            data_GT = self.transform(gt_img)
            data_MASK = self.transform(mask_img)

        _, h, w = data_GT.shape
        if (h % self.factor != 0) or (w % self.factor != 0):
            data_GT = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_GT)
            data_IN = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_IN)
            data_MASK = transforms.Resize(((h // self.factor) * self.factor, (w // self.factor) * self.factor))(data_MASK)

        return data_IN, data_GT, data_MASK, img_name

    def __len__(self):
        return len(self.imgs_in)

class my_datasetY(Dataset):
    def __init__(self,root_in,root_label,transform =None):
        super(my_datasetY,self).__init__()
        #in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)

        if self.transform:
            in_img = np.expand_dims(np.asarray(in_img),-1)
            data_IN = self.transform(in_img)
            gt_img = np.expand_dims(np.asarray(gt_img), -1)
            data_GT = self.transform(gt_img)
        else:
            data_IN =np.expand_dims(np.asarray(in_img)/255.0,0)
            data_IN = torch.as_tensor(data_IN,torch.float32)

            data_GT = np.expand_dims(np.asarray(gt_img)/255.0)
            data_GT = torch.as_tensor(data_GT,torch.float32)
        return data_IN,data_GT
    def __len__(self):
        return len(self.imgs_in)

if __name__ == '__main__':
    in_root = '/gdata2/zhuyr/ShadowD/Datasets' #'D:/Research/Shadow'
    gt_root =  '/ghome/zhuyr/ILR/datasets/AISTD-train-p8_common.txt'

    train_set = my_dataset_threeIn_wAug(in_root, gt_root, Crop = True, crop_size = 256, factor = 8, sample= True,fix_sample =100000,Aug=True)
    train_loader = DataLoader(train_set, batch_size= 1, num_workers=4, shuffle=True, drop_last=False,pin_memory=True)
    #train_loader +
    print('len(train_loader):',len(train_loader))
    import tqdm
    for train_idx, (data_in, label, mask) in tqdm.tqdm(enumerate(train_loader, 0),total=len(train_loader)):
        if train_idx% 20 ==0:
            print('---------' * 2, train_idx)
            print(data_in.size(), label.size(), mask.size())