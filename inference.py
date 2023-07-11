import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import workspace.qingguangyan.resnet_medicalnet as resnet_medicalnet
import transforms as trans
import dataset
from workspace.qingguangyan.resnet import *
from workspace.qingguangyan.resnet_3d import *
import warnings
warnings.filterwarnings('ignore')

image_size = 224
oct_img_size = [448, 448]
testset_root = '/home/xxxxxx/data/Glaucoma_grading/testing/multi-modality_images'
best_model_path = "/home/xxxxxx/workspace/qingguangyan/best_model/0315_GAMMA_Med_0.6377.pth"
device = "cuda:0"

# model
# backbone_oct = resnet18(pretrained=False)
# backbone_oct.conv1 = nn.Conv2d(256, 64,
#                         kernel_size=7,
#                         stride=2,
#                         padding=3,
#                         bias=False)

# backbone_oct = mc3_18(pretrained=True)
# backbone_oct.stem[0] = nn.Conv3d(256, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
#                                  padding=(1, 3, 3), bias=False)

backbone_oct = resnet_medicalnet.resnet18(pretrained=False, shortcut_type="A")
backbone_oct.conv1 = nn.Conv3d(
    256,
    64,
    kernel_size=7,
    stride=(2, 2, 2),
    padding=(3, 3, 3),
    bias=False)


backbone_fundus = resnet18(pretrained=False)
classifier = nn.Linear(512 * 2, 3)

para_state_dict = torch.load(best_model_path, map_location="cpu")

backbone_fundus.load_state_dict(para_state_dict["backbone_fundus"])
backbone_oct.load_state_dict(para_state_dict["backbone_oct"])
classifier.load_state_dict(para_state_dict["classifier"])

backbone_fundus.eval()
backbone_oct.eval()
classifier.eval()

backbone_fundus.to(device)
backbone_oct.to(device)
classifier.to(device)

oct_test_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])
img_test_transforms = trans.Compose([
    trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

test_dataset = dataset.GAMMA_sub1_dataset(dataset_root=testset_root,
                                  img_transforms=img_test_transforms,
                                  oct_transforms=oct_test_transforms,
                                  mode='test')

cache = []
for fundus_img, oct_img, idx in tqdm(test_dataset):

    fundus_img = fundus_img[np.newaxis, ...]
    oct_img = oct_img[np.newaxis, ...]

    fundus_img = torch.tensor(fundus_img / 255, dtype = torch.float32).to(device)
    # oct_img = torch.tensor(oct_img / 255, dtype = torch.float32).to(device)
    oct_img = torch.tensor(oct_img, dtype = torch.float32).unsqueeze(2).to(device)


    logits_fundus = backbone_fundus(fundus_img)
    logits_fundus = F.adaptive_avg_pool2d(logits_fundus, 1).flatten(1)
    logits_oct = backbone_oct(oct_img)
    logits_oct = F.adaptive_avg_pool2d(logits_oct, 1).flatten(1)
    logits = torch.cat([logits_fundus, logits_oct], dim=1)
    scores = classifier(logits)
    
    cache.append([idx, scores.detach().cpu().numpy().argmax()])

submission_result = pd.DataFrame(cache, columns=['data', 'dense_pred'])
submission_result['non'] = submission_result['dense_pred'].apply(lambda x: int(x == 0))
submission_result['early'] = submission_result['dense_pred'].apply(lambda x: int(x == 1))
submission_result['mid_advanced'] = submission_result['dense_pred'].apply(lambda x: int(x == 2))
submission_result[['data', 'non', 'early', 'mid_advanced']].to_csv("/home/xxxxxx/workspace/qingguangyan/Results_0315_Med18_224.csv", index=False)
