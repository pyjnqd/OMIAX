import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *

from tqdm import tqdm
from dataset import GAMMA_dataset
import resnet, resnet_medicalnet, resnet_3d

import warnings
warnings.simplefilter("ignore")


# ===================================== 超参数 ===================================== #

batchsize = 8  
oct_img_size = 256
epochs = 20
# trainset_root = '/dev/shm/GAMMA/Train/multi-modality_images'
# testset_root = '/dev/shm/GAMMA/Test/multi-modality_images'
trainset_root = '/dev/shm/hospital/trainval'
testset_root = '/dev/shm/hospital/test'
num_workers = 8
backbone_lr = 1e-4
head_lr = 1e-3
data_source = 1 # 0: fundus 1: oct 2: fundus+oct 
exp_name = "0701-100-100-vertical"

# run = wandb.init(
#     project="qingguangyan-875",
#     name=exp_name,
#     notes="baseline on oct only split",
#     config={
#         "backbone_lr": backbone_lr,
#         "head_lr": head_lr,
#         "epochs": epochs,
#         "batchsize": batchsize,
#         "oct_img_size": oct_img_size,
#     })


# ===================================== 数据 ===================================== #

print(f"\033[32m Note: We regard test dataset as validation set and unify the two phase into one. \033[0m")

train_dataset = GAMMA_dataset(dataset_root=trainset_root,
                            fundus_size=None,
                            oct_size=oct_img_size,
                            label_file=osp.join(trainset_root, '../training_GT.xlsx'),
                            mode='train',
                            data_source = data_source) 

val_dataset = GAMMA_dataset(dataset_root=testset_root,
                            fundus_size=None,
                            oct_size=oct_img_size,
                            label_file=osp.join(testset_root, '../testing_GT.xlsx'),
                            mode='val',
                            data_source = data_source) 

train_loader = DataLoader(train_dataset, batch_size = batchsize, num_workers = num_workers, pin_memory = True)
val_loader = DataLoader(val_dataset, batch_size = batchsize, num_workers = num_workers, pin_memory = True)

# ===================================== 模型及优化 ===================================== #

backbone_oct = resnet.resnet18(pretrained=True)
backbone_oct.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
# backbone_oct = resnet_3d.r3d_18(pretrained=True)
# backbone_oct.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
# backbone_oct = resnet_3d.r2plus1d_18(pretrained=True)
# backbone_oct.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
backbone_oct = backbone_oct.cuda()
# classifier = nn.Linear(512, 3).cuda()
classifier = nn.Linear(512, 4).cuda()

params_group =[
    {"params": backbone_oct.parameters(), 'lr':backbone_lr},
    {"params": classifier.parameters(), 'lr':head_lr}
]

optimizer = torch.optim.Adam(params=params_group)
# optimizer = torch.optim.SGD(params=params_group)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

criterion = nn.CrossEntropyLoss()




# ========================================== VAL =========================================== #

def val(backbone_oct, classifier, val_dataloader, criterion):
    backbone_oct.eval()
    classifier.eval()
    avg_val_loss = Averager()
    cache = []
    with torch.no_grad():
        for data in (bar := tqdm(val_dataloader)):
            bar.set_description(f"Doing validation: ")
            # oct_imgs = data[0].unsqueeze(1).cuda() # for resnet_3d
            oct_imgs = data[0].cuda()
            labels = data[1].cuda()

            logits = backbone_oct(oct_imgs)
            if logits.dim() == 4:
                logits = F.adaptive_avg_pool2d(logits, 1).flatten(1)
            scores = classifier(logits)

            for p, l in zip(scores.detach().cpu().numpy().argmax(1), labels.cpu().numpy()):
                cache.append([p, l])

            loss = criterion(scores, labels)
            avg_val_loss.add(loss)

    backbone_oct.train()
    classifier.train()
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    acc = accuracy_score(cache[:, 1], cache[:, 0], normalize=True) 
    matrix = confusion_matrix(cache[:, 1], cache[:, 0], normalize="true")
    matrix = matrix.diagonal()
    avg_loss = avg_val_loss.item()

    return avg_loss, kappa, acc, matrix

# ========================================= TRAIN ========================================== #
backbone_oct.train()
classifier.train()
avg_train_loss = Averager()
cache = []
best_kappa = 0.
max_val_kappa = -1
for epoch in range(epochs):
    for i, data in enumerate(bar := tqdm(train_loader)):
        torch.cuda.empty_cache()
        bar.set_description(f"Epoch: {epoch} ITER: {i}")
        optimizer.zero_grad()
        # import pdb;pdb.set_trace()
        # oct_imgs = data[0].unsqueeze(1).cuda() # for resnet_3d
        oct_imgs = data[0].cuda() 
        labels = data[1].cuda()

        logits = backbone_oct(oct_imgs)
        if logits.dim() == 4:
            logits = F.adaptive_avg_pool2d(logits, 1).flatten(1)
        scores = classifier(logits)

        loss = criterion(scores, labels)

        for p, l in zip(scores.detach().cpu().numpy().argmax(1), labels.cpu().numpy()):
            cache.append([p, l])

        loss.backward()
        optimizer.step()
        avg_train_loss.add(loss)

        # val_loss, val_kappa, val_acc, matrix = val(backbone_oct, classifier, val_loader, criterion)
        # print("[EVAL] epoch={}/{} val_loss={:.4f} val_kappa={:.4f} val_acc={:.4f} val_non_acc={:.4f} val_early_acc={:.4f} val_mid&adv_acc={:.4f}".format(
        #     epoch, epochs, val_loss, val_kappa, val_acc, matrix[0], matrix[1], matrix[2]))
        # if val_kappa > max_val_kappa:
        #     max_val_kappa = val_kappa
    scheduler.step()

    # log once
    cache = np.array(cache)
    train_kappa = cohen_kappa_score(cache[:, 1], cache[:, 0], weights='quadratic')
    print("[TRAIN] epoch={}/{} train_loss={:.4f} train_kappa={:.4f}".format(epoch, epochs, avg_train_loss.item(), train_kappa))
    # wandb.log({"epoch":epoch, "train_loss": avg_train_loss.item(), "train_kappa": train_kappa})
    cache = []
    avg_train_loss.reset()
    

    # val once
    val_loss, val_kappa, val_acc, matrix = val(backbone_oct, classifier, val_loader, criterion)
    # wandb.log({"epoch":epoch, "val_loss": val_loss, "val_kappa": val_kappa, "val_acc": val_acc, 
    #            "val_non_acc": matrix[0], "val_early_acc": matrix[1], "val_mid&adv_acc": matrix[2]})
    print("[EVAL] epoch={}/{} val_loss={:.4f} val_kappa={:.4f} val_acc={:.4f} val_non_acc={:.4f} val_early_acc={:.4f} val_mid&adv_acc={:.4f}".format(
        epoch, epochs, val_loss, val_kappa, val_acc, matrix[0], matrix[1], matrix[2]))
    
    # if val_kappa > best_kappa:
    #     if best_kappa != 0.:
    #         os.system("rm -f /home/xxxxxx/workspace/qingguangyan/best_model/oct/{0}_{1:.4f}.pth".format(exp_name, best_kappa))
    #     best_kappa = val_kappa
    #     torch.save({"backbone_oct": backbone_oct.state_dict(), 
    #                 "classifier": classifier.state_dict()},
    #                 os.path.join("/home/xxxxxx/workspace/qingguangyan/best_model/oct", "{0}_{1:.4f}.pth".format(exp_name, best_kappa)))
    

print(max_val_kappa)


