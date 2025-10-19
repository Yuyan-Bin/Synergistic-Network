import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# from torch.testing._internal.common_utils import SEED
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import logging
import os
import torch.nn as nn
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import optim
import random
import time
import albumentations as A
# from albumentations.pytorch import ToTensor
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from loader import  binary_class
# from loader import  binary_class2
from sklearn.model_selection import GroupKFold
# from loss import *
# from loss import DiceLoss_binary
# from loss import  IoU_binary

# from synapse_train_test.networks.bra_unet import BRAUnet
from networks.bra_unet import BRAUnet
from thop import profile
# *************************loss.py********************************
import torch.nn as nn
import torch
# from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np


# cfs = ConfusionMatrix(3)
class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    torch.nn.BCEWithLogitsLoss


class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


# **********************************************************************

# **********************************************************************
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# **********************************************************************

def cal_params_flops(model, size):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)            ## 打印计算量
    print('params',params/1e6)            ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    # 返回 FLOPs 和参数数量
    return flops, params, total

def get_train_transform():
    return A.Compose(
        [
            # A.Resize(256, 256),
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0, p=0.25),
            A.CoarseDropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


def get_valid_transform():
    return A.Compose(
        [
            # A.Resize(256, 256),
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}

    best_model_wts = model.state_dict()
    best_loss = float('inf')
    counter = 0

    # ***********************************************************************************

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d_%H%M')
    # ************************************************************************************
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)


            running_loss = []
            running_corrects = []

            # Iterate over data
            # for inputs,labels,label_for_ce,image_id in dataloaders[phase]:
            for inputs, labels, image_id in dataloaders[phase]:
                # wrap them in Variable
                if torch.cuda.is_available():

                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    # label_for_ce = Variable(label_for_ce.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                score = accuracy_metric(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects.append(score.item())

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)

            print('{} Loss: {:.4f} IoU: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            Loss_list[phase].append(epoch_loss)
            Accuracy_list[ phase].append(epoch_acc)
            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                # ************************************************************************
                logging.info(
                    'Epoch [{}/{}] - Validation Loss: {:.4f}, IoU: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss,
                                                                                  epoch_acc))
                # ************************************************************************
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                counter = 0
                # if epoch > 0:
                if epoch > 75:
                    # torch.save(model.state_dict(), f'save_models/epoch_{epoch}_{epoch_acc}_.pth')
                    torch.save(model.state_dict(), f'save_models/epoch_{epoch}_bs_{args.batch}_{formatted_time}_acc_{epoch_acc}_.pth',_use_new_zipfile_serialization=False)
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'train':
                print('Current learning rate:', optimizer.param_groups[0]['lr'])
                scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, Loss_list, Accuracy_list
if __name__ == '__main__':
    # *************************************
    log_file = 'training_log.txt'
    setup_logging(log_file)
    # *************************************
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动调优功能，以确保每次运行的结果一致。
    os.environ['PYTHONHASHSEED'] = str(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:",device)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='/home/cqut/Data/medical_seg_data/ISIC2018_jpg/', help='the path of images')
    parser.add_argument('--dataset', type=str, default='/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/data', help='the path of images')
    # parser.add_argument('--csvfile', type=str, default='src/test_train_data.csv',
    #                     help='two columns [image_id,category(train/test)]rrf')
    parser.add_argument('--csvfile', type=str, default='/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/test_train_data.csv',
                        help='two columns [image_id,category(train/test)]')
    parser.add_argument('--loss', default='dice', help='loss type')
    # parser.add_argument('--loss', default='ce', help='loss type')
    parser.add_argument('--batch', type=int, default=44, help='batch size') #自己模型的迁移参数
    # parser.add_argument('--batch', type=int, default=16, help='batch size')#BRAUnet++迁移参数（按照论文中的16）
    parser.add_argument('--lr', type=float, default=0.0006, help='learning rate') #自己模型的迁移参数
    # parser.add_argument('--lr', type=float, default=0.0005, help='learning rate') #BRAUnet++迁移参数
    parser.add_argument('--epoch', type=int, default=180, help='epoches')#自己模型的迁移参数
    # parser.add_argument('--epoch', type=int, default=200, help='epoches')#BRAUnet++迁移参数
    args = parser.parse_args()
    os.makedirs(f'save_models/', exist_ok=True)
    df = pd.read_csv(args.csvfile)
    df = df[df.category == 'train'] # 筛选出类别为train的数据行，即只保留用于训练的数据。
    df.reset_index(drop=True, inplace=True)     # 重置DataFrame的索引
    # gkf = GroupKFold(n_splits=5) #BRAUnet++迁移参数
    gkf = GroupKFold(n_splits=9)  #自己模型迁移参数
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df.image_id.tolist())):
        df.loc[val_idx, 'fold'] = fold
    # **************************************************
     # 打印每一折的训练集和验证集数量
    print("Total number of training samples:", len(df))
    for fold in range(9):
        train_df = df[df.fold != fold]
        val_df = df[df.fold == fold]
        print(f"Fold {fold}:")
        print(f"  Training set size: {len(train_df)}")
        print(f"  Validation set size: {len(val_df)}")
        print(f"  Total for this fold: {len(train_df) + len(val_df)}")
        print(f"  Percentage used for validation: {len(val_df) / len(df) * 100:.2f}%")
        print(f"  Percentage used for training: {len(train_df) / len(df) * 100:.2f}%")
        print()

    # 示例：选择第0折进
    # ****************************************************
    fold = 0
    val_files = list(df[df.fold == fold].image_id)
    print("Validation files for fold", fold, ":", val_files)
    print("Number of validation files:", len(val_files))
    train_files = list(df[df.fold != fold].image_id)
    print("Training files for fold", fold, ":", train_files[:5], "...")
    print("Number of training files:", len(train_files))

    train_dataset = binary_class(args.dataset, train_files, get_train_transform())
    val_dataset = binary_class(args.dataset, val_files, get_valid_transform())

    # train_dataset = binary_class2(args.dataset, train_files, get_train_transform())
    # val_dataset = binary_class2(args.dataset, val_files, get_valid_transform())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,
                                               drop_last=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch//4, drop_last=True,num_workers=8)
    dataloaders = {'train': train_loader, 'valid': val_loader}

    # model_ft = BRAUnet(img_size=256,in_chans=3, num_classes=1, n_win=8)
    model_ft = BRAUnet(img_size=224,in_chans=3, num_classes=1, n_win=7)
    model_ft.load_from()
    # # **************************************************
    #
    # randn_input = torch.randn(1, 3, 256, 256)
    # flops, params = profile(model_ft, inputs=(randn_input,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # # **************************************************
    if torch.cuda.is_available():
        model_ft = model_ft.cuda()
    # # *******************************************************************************************
    # 计算参数数量
    # 创建输入数据，并确保它也在GPU上
    # input = torch.randn(1, 3, 256, 256).to('cuda')
    input = torch.randn(1, 3, 224, 224).to('cuda')
    # 现在调用计算参数和浮点运算次数的函数
    # # print(cal_params_flops(model_ft, 256))
    print(cal_params_flops(model_ft, 224))
    # # *******************************************************************************************
    # Loss, IoU and Optimizer
    if args.loss == 'ce':
        criterion = nn.BCELoss()
    if args.loss == 'dice':
        criterion = DiceLoss_binary()

    accuracy_metric = IoU_binary()



# ************************************************************************************************************
#     optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr) #BRAUNet++的默认设置
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.lr) #自己模型迁移设置
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 200, eta_min=0, last_epoch=-1, verbose=False)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, 200, eta_min=0, last_epoch=-1, verbose=False)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, 200, eta_min=0, last_epoch=-1, verbose=False)

    model_ft, Loss_list, Accuracy_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                     num_epochs=args.epoch)

    # ***********************************************************************************

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d_%H%M')
    # ****************************************************************************
    torch.save(model_ft.state_dict(), f'save_models/epoch_last_{formatted_time}.pth',_use_new_zipfile_serialization=False)
    # ****************************************************************************
    # 绘制损失曲线图
    epochs = range(1, args.epoch + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, Loss_list['train'], 'r', label='Training Loss')
    plt.plot(epochs, Loss_list['valid'], 'b', label='Validation Loss')
    plt.title(f'Training and Validation Loss_epoch_{args.epoch}_bs_{args.batch}_{formatted_time}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, Accuracy_list['train'], 'r', label='Training IoU')
    plt.plot(epochs, Accuracy_list['valid'], 'b', label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_and_iou_plot.png')
    plt.show()




