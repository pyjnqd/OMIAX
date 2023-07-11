from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import resnet, resnet_medicalnet
import torch
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import numpy as np

txt_path = "/home/xxxxxx/workspace/OMIAX/vertical.txt"
oct_range = {}
with open(txt_path, "r") as f:  
    for line in f.readlines():
        line = line.strip('\n')
        data = line.split(" ")
        oct_range[data[0]] = int(data[1])

oct_transforms = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ids = ["9947","9978","10160","7288","31536","33373","11140","36271","34278"]
# id = "0014"
for id in ids:
    model = resnet.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False)

    state_dict = torch.load("/home/xxxxxx/workspace/OMIAX/save/hospital_128_0.7311.pth", map_location="cpu")
    model.load_state_dict(state_dict["backbone_oct"], strict=False)
    model.classifier.load_state_dict(state_dict["classifier"])
    target_layers = [model.layer4[-1]]

    oct_path = f"/home/xxxxxx/data/hospital/test/{id}/oct"
    # oct_path = f"/home/xxxxxx/data/GAMMA/Train/multi-modality_images/{id}/{id}"
    oct_series_list = sorted(os.listdir(oct_path), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    oct_series_list = oct_series_list[::2]
    oct_img = []
    for oct_image_name in tqdm(oct_series_list):
        img = Image.open(os.path.join(oct_path, oct_image_name)).convert('L')
        img = img.crop((0, oct_range[id], 512, oct_range[id]+320))
        img = oct_transforms(img)
        oct_img.append(img.squeeze_(0))
    oct_img = torch.stack(oct_img) 
    oct_img = oct_img.unsqueeze(0)# 1 1 D 256 256

    # oct_path = f"/home/xxxxxx/data/hospital/test/{id}/vertical"
    # # oct_path = f"/home/xxxxxx/data/GAMMA/Train/multi-modality_images/{id}/oct"
    # oct_series_vertical_list = sorted(os.listdir(oct_path), key=lambda x: int(x.split(".")[0]))
    # oct_img = []
    # for oct_image_name in tqdm(oct_series_vertical_list):
    #     img_v = Image.open(os.path.join(oct_path, oct_image_name)).convert('L')
    #     img_v = oct_transforms(img_v)
    #     oct_img.append(img_v.squeeze_(0))
    # oct_img = torch.stack(oct_img)
    # oct_img = oct_img.unsqueeze(0)

    print(model(oct_img))
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # targets = [ClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=oct_img, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # os.makedirs(f"test/{id}")
    for oct_image_name in tqdm(oct_series_list):
        img = Image.open(os.path.join(oct_path, oct_image_name)).convert('RGB')
        img = img.crop((0, oct_range[id], 512, oct_range[id]+320))
        img = img.crop((128, 32, 384, 288))
        img = np.array(img)/255.0
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False, image_weight=0.9)
        Image.fromarray(visualization).save(f"test/{id}/{oct_image_name}")

    # for oct_image_name in tqdm(oct_series_vertical_list):
    #     img_v = Image.open(os.path.join(oct_path, oct_image_name)).convert('RGB')
    #     img_v = img_v.crop((128, 0, 384, 256))
    #     img_v = np.array(img_v)/255.0
    #     visualization = show_cam_on_image(img_v, grayscale_cam, use_rgb=False, image_weight=0.85)
    #     Image.fromarray(visualization).save(f"test/{id}/v_{oct_image_name}")
