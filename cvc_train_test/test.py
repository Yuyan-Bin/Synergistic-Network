import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from loader import binary_class
import albumentations as A
from albumentations.pytorch import ToTensor
# from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import BinaryAccuracy,BinaryPrecision,BinaryRecall,BinaryF1Score
import torchmetrics
import argparse
import time
import pandas as pd
import cv2
import random
import os
from networks.bra_unet import BRAUnet
import tifffile as tf
import rasterio
from matplotlib import pyplot as plt
import tifffile
from PIL import Image

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # print("inputs111kkk", inputs.shape)
        inputs = inputs.view(-1)
        targets = targets.view(-1)


        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

def get_transform():
   return A.Compose(
       [
        A.Resize(256,256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='/home/cqut/Data/medical_seg_data/ISIC2018_jpg/',type=str, help='the path of dataset')
    parser.add_argument('--dataset', default='/tmp/pycharm_project_794/BRAU-Netplusplus-master/cvc_train_test/data',type=str, help='the path of dataset')
    # parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,
    # category(train/test)]')
    parser.add_argument('--csvfile', default='/tmp/pycharm_project_794/BRAU-Netplusplus-master/cvc_train_test/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model',default='save_models/epoch_last_20250113_1343.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    os.makedirs('debug/',exist_ok=True)
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    print(test_files)
    print(len(test_files))

    test_dataset = binary_class(args.dataset,test_files, get_transform())
    model_ft = BRAUnet(img_size=256,in_chans=3, num_classes=1, n_win=8)

    model_w = torch.load(args.model)
    # model_ft.load_state_dict(model_w)
    model_ft.load_state_dict(model_w,False)
    model = model_ft.cuda()
    # acc_eval = Accuracy()
    # acc_eval = torchmetrics.Accuracy()
    acc_eval = BinaryAccuracy()
    # pre_eval = Precision()
    pre_eval = BinaryPrecision()
    dice_eval = Dice()
    # recall_eval = Recall()
    recall_eval = BinaryRecall()
    # f1_eval = F1(2)
    # f1_eval = F1Score(2)
    f1_eval = BinaryF1Score()
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    since = time.time()
    for image_id in test_files:

        # # *******************修改*********************************
        img_path = f'/tmp/pycharm_project_794/BRAU-Netplusplus-master/cvc_train_test/data/images/{image_id}'
        orig_img= tifffile.imread(img_path)

        if len( orig_img.shape) == 2:  # 如果是灰度图，则转换为三通道
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        # *********************************************************
        # if img is None:
        #     print(f"Error: Failed to load image {img_path}")
        #     continue  # Skip this iteration and move on to the next image.
        orig_img = cv2.resize(orig_img, (256, 256))
        img_id = list(image_id.split('.'))[0]
        # cv2.imwrite(f'debug/{img_id}.tif',img)
        Image.fromarray(orig_img).save(f'debug/{img_id}.tif')

    model.eval()
    with torch.no_grad():
        for img, mask, img_id in test_dataset:
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad =False).cuda()
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)  # 将图像传递给模型进行预测，得到输出pred
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)  
            pred = torch.sigmoid(pred)  
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0   
            pred_draw = pred.clone().detach()   
            mask_draw = mask.clone().detach()
            if args.debug:
                img_id = list(img_id.split('.'))[0]
                # ***********************xiugai************************************
                # Save prediction and ground truth masks as .tif files with color information preserved.
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                img_numpy[img_numpy == 1] = 255
                tifffile.imwrite(f'debug/{img_id}_pred.tif', img_numpy)

                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy == 1] = 255
                tifffile.imwrite(f'debug/{img_id}_gt.tif', mask_numpy)

                # Load the original .tif image while preserving its color mode.
                orig_img_path = f'/tmp/pycharm_project_794/BRAU-Netplusplus-master/cvc_train_test/data/images/{img_id}.tif'

                # Check if the file exists and is readable
                if not os.path.isfile(orig_img_path) or not os.access(orig_img_path, os.R_OK):
                    print(f"Warning: Image {orig_img_path} does not exist or is not readable.")
                    continue

                try:
                    orig_img = tifffile.imread(orig_img_path)
                except Exception as e:
                    print(f"Warning: Failed to load image {orig_img_path}: {e}")
                    continue

                # Ensure the image was loaded successfully and has multiple channels (color image)
                if len(orig_img.shape) == 2:  # If it's a grayscale image, convert it to BGR
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                orig_img_resized = cv2.resize(orig_img, (256, 256), interpolation=cv2.INTER_LINEAR)

                # Create a copy of the original image to draw contours on without modifying the original one.
                orig_img_with_contours =orig_img_resized.copy()

                # Convert prediction and mask to binary masks for contour detection.
                pred_binary = pred_draw.cpu().detach().numpy()[0][0].astype(np.uint8)
                mask_binary = mask_draw.cpu().detach().numpy()[0][0].astype(np.uint8)

                # Find and draw contours for ground truth (green color).
                contours_gt, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_img_with_contours, contours_gt, -1, ((0, 180, 0)), 1)

                # Find and draw contours for prediction (blue color).
                contours_pred, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_img_with_contours, contours_pred, -1, (0, 0, 255), 1)  # Blue color.

                # Save the resulting image with contours using tifffile to preserve quality.
                contours_save_path = f'debug/{img_id}_contours.tif'
                tifffile.imwrite(contours_save_path, orig_img_with_contours, photometric='rgb')
                print(f"Saved contours image to {contours_save_path}")
            iouscore = iou_eval(pred, mask)
            dicescore = dice_eval(pred, mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
            # accscore = acc_eval(pred.cpu(),mask.cpu())
            accscore = acc_eval(pred.cpu(), mask.cpu().long())
            # prescore = pre_eval(pred.cpu(),mask.cpu())
            prescore = pre_eval(pred.cpu(),mask.cpu().long())
            # recallscore = recall_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(),mask.cpu().long())
            # f1score = f1_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(),mask.cpu().long())
            iou_score.append(iouscore.cpu().detach().numpy())
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    # **************************************************************************
    # 增添的部分：将输出结果保存到 test_log.txt 文件中
    log_dir = 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'test_log.txt')
    with open('test_log.txt', 'w') as log_file:
        log_file.write('Evaluation complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        log_file.write('FPS: {:.2f}\n'.format(1.0 / (sum(time_cost) / len(time_cost))))
        log_file.write('mean IoU: {:.4f} {:.4f}\n'.format(np.mean(iou_score), np.std(iou_score)))
        log_file.write('mean dice: {:.4f} {:.4f}\n'.format(np.mean(dice_score), np.std(dice_score)))
        log_file.write('mean accuracy: {:.4f} {:.4f}\n'.format(np.mean(acc_score), np.std(acc_score)))
        log_file.write('mean precision: {:.4f} {:.4f}\n'.format(np.mean(pre_score), np.std(pre_score)))
        log_file.write('mean recall: {:.4f} {:.4f}\n'.format(np.mean(recall_score), np.std(recall_score)))
        log_file.write('mean F1-score: {:.4f} {:.4f}\n'.format(np.mean(f1_score), np.std(f1_score)))
# ***********************************************************************************
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean IoU:',round(np.mean(iou_score),4),round(np.std(iou_score),4))
    print('mean dice:',round(np.mean(dice_score),4),round(np.std(dice_score),4))
    print('mean accuracy:',round(np.mean(acc_score),4),round(np.std(acc_score),4))
    print('mean precsion:',round(np.mean(pre_score),4),round(np.std(pre_score),4))
    print('mean recall:',round(np.mean(recall_score),4),round(np.std(recall_score),4))
    print('mean F1-score:',round(np.mean(f1_score),4),round(np.std(f1_score),4))
