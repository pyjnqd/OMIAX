import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
# import pyvips
import numpy as np
# testing one image broken
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

txt_path = "/home/xxxxxx/workspace/OMIAX/vertical.txt"
# txt_path = "/home/xxxxxx/workspace/OMIAX/hospital.txt"
oct_range = {}
with open(txt_path, "r") as f:  
    for line in f.readlines():
        line = line.strip('\n')
        data = line.split(" ")
        oct_range[data[0]] = int(data[1])

class GAMMA_dataset(Dataset):
    """
    getitem() output:
    
    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)
        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 fundus_size,
                 oct_size,
                 dataset_root,
                 mode,
                 label_file,
                 data_source = None):
        
        self.dataset_root = dataset_root
        self.mode = mode.lower()
        self.data_source = data_source

        if self.mode == 'train':
            label = {row['data']:row[1:].values
                        for _,row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == 'val':
            label = {row['data']:row[1:].values
                        for _,row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]


        if mode == "train":
            if fundus_size is not None:
                self.img_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(
                        fundus_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)), # scale and ratio need to tune carefully
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(30),
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    transforms.Normalize([0.2593434453010559, 0.128742977976799, 0.038363877683877945], 
                                          [0.2910853624343872, 0.15626344084739685, 0.0723380446434021])
                ])
            if oct_size is not None:
                self.oct_transforms = transforms.Compose([
                    transforms.CenterCrop(oct_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
        else:
            if fundus_size is not None:
                self.img_transforms = transforms.Compose([
                    transforms.Resize([fundus_size, fundus_size]),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    transforms.Normalize([0.2593434453010559, 0.128742977976799, 0.038363877683877945], 
                                          [0.2910853624343872, 0.15626344084739685, 0.0723380446434021])
                ])
            if oct_size is not None:
                self.oct_transforms = transforms.Compose([
                    transforms.CenterCrop(oct_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        label = torch.LongTensor(label)
        label = label.argmax()

        if self.data_source == 0 or self.data_source == 2:
            fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
            fundus_img = self.img_transforms(Image.open(fundus_img_path).convert('RGB'))

        if self.data_source == 1 or self.data_source == 2:
            # oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
            #                     key=lambda x: int(x.split("_")[0]))
            # # oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, "oct")),
            # #         key=lambda x: int(x.split("_")[-1].split(".")[0]))
            # oct_series_list = oct_series_list[::1]
            # oct_img = []
            # for oct_image_name in oct_series_list:
            #     img = Image.open(os.path.join(self.dataset_root, real_index, real_index, oct_image_name)).convert('L')
            #     # img = Image.open(os.path.join(self.dataset_root, real_index, "oct", oct_image_name)).convert('L')
            #     # img = img.crop((0, oct_range[real_index], 512, oct_range[real_index]+320))
            #     img = self.oct_transforms(img)
            #     oct_img.append(img.squeeze_(0))
            # oct_img = torch.stack(oct_img)


            oct_series_vertical_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, "vertical")),
                    key=lambda x: int(x.split(".")[0]))
            oct_img = []
            for oct_image_name in oct_series_vertical_list:
                img_v = Image.open(os.path.join(self.dataset_root, real_index, "vertical", oct_image_name)).convert('L')
                img_v = self.oct_transforms(img_v)
                oct_img.append(img_v.squeeze_(0))
            oct_img = torch.stack(oct_img)
            
        
            

        if self.data_source == 0:
            return fundus_img, label
        elif self.data_source == 1:
            return oct_img, label
        elif self.data_source == 2:
            return fundus_img, oct_img, label
        else:
            raise Exception("data source is unknown.")

    def __len__(self):
        return len(self.file_list)
    


""" GAMMA NORMALIZE PARAMETER: MEAN & STD
WITH RESIZE 256^2 
R: 0.2617436945438385 0.29011139273643494
G: 0.12980514764785767 0.15516449511051178
B: 0.04017869383096695 0.0727657750248909

WITH RESIZE 512^2 
R: 0.26174989342689514 0.2904183864593506
G: 0.12978801131248474 0.1554831564426422
B: 0.040165215730667114 0.07302083820104599

ONLY WITH TOTENSOR()
R: 0.2593434453010559 0.2910853624343872
G: 0.128742977976799 0.15626344084739685
B: 0.038363877683877945 0.0723380446434021
[(0.2593434453010559, 0.128742977976799, 0.038363877683877945), 
 (0.2910853624343872, 0.15626344084739685, 0.0723380446434021)]

CENTERCROP 512^2
channel 0: 0.311550110578537 0.09565852582454681
"""