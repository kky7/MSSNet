import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, image_list, root_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        datalist = open(image_list, 'r')
        self.image_files = datalist.readlines()
        self.root_dir = root_dir

        self.img_options = img_options
        self.sizex       = len(self.image_files)

        self.ps = self.img_options['patch_size']
        print('trainset num:',self.sizex)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        ps = self.ps

        images_dir = self.image_files[index][0:-1]
        split_images_dir = images_dir.split(' ')
        sharp_image_dir = split_images_dir[0]
        blur_image_dir = split_images_dir[1]

        inp_img = Image.open(os.path.join(self.root_dir, blur_image_dir)).convert('RGB')
        tar_img = Image.open(os.path.join(self.root_dir, sharp_image_dir)).convert('RGB')

        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        aug    = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        aug    = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))

        return tar_img, inp_img

class DataLoaderVal(Dataset):
    def __init__(self, image_list, root_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        datalist = open(image_list, 'r')
        self.image_files = datalist.readlines()
        self.root_dir = root_dir

        self.img_options = img_options
        self.sizex       = len(self.image_files)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        ps = self.ps

        images_dir = self.image_files[index][0:-1]
        split_images_dir = images_dir.split(' ')
        sharp_image_dir = split_images_dir[0]
        blur_image_dir = split_images_dir[1]

        tar_img = Image.open(os.path.join(self.root_dir, sharp_image_dir)).convert('RGB')
        inp_img = Image.open(os.path.join(self.root_dir, blur_image_dir)).convert('RGB')

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        return tar_img, inp_img

class DataLoaderTest(Dataset):
    def __init__(self, image_list, root_dir):
        super(DataLoaderTest, self).__init__()

        datalist = open(image_list, 'r')
        self.image_files = datalist.readlines()
        self.root_dir = root_dir

        self.sizex       = len(self.image_files)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        images_dir = self.image_files[index][0:-1]
        split_images_dir = images_dir.split(' ')
        sharp_image_dir = split_images_dir[0]
        blur_image_dir = split_images_dir[1]

        inp_img = Image.open(os.path.join(self.root_dir, blur_image_dir)).convert('RGB')
        tar_img = Image.open(os.path.join(self.root_dir, sharp_image_dir)).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        blur_name = blur_image_dir.split('/')[-1]

        return tar_img, inp_img, blur_name
