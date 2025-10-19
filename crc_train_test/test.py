import os
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd95

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torchmetrics
import argparse
import time
import pandas as pd
import cv2
import random
import os
from networks.bra_unet import BRAUnet
# *****************增加dice和hd95计算**************************
# def dice_coefficient(pred, target):
#     intersection = (pred * target).sum()
#     dice = (2. * intersection) / (pred.sum() + target.sum())
#     return dice

def hausdorff_distance_95(pred, target):
    pred_np = pred.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)
    hd = hd95(pred_np, target_np)
    return hd
# ***********************************************************
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
        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


def get_transform():
    return A.Compose(
        [
            # A.Resize(256, 256),
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='/home/cqut/Data/medical_seg_data/ISIC2018_jpg/',type=str, help='the path of dataset')
    parser.add_argument('--dataset',
                        default='/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/data', type=str,
                        help='the path of dataset')
    # parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--csvfile',
                        default='/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/test_train_data.csv',
                        type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model', default='save_models/epoch_last_20250417_0801.pth', type=str,
                        help='the path of model')
    parser.add_argument('--debug', default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    os.makedirs('debug/', exist_ok=True)
    df = pd.read_csv(args.csvfile)
    df = df[df.category == 'test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    print(test_files)
    print(len(test_files))
    test_dataset = binary_class(args.dataset, test_files, get_transform())
    # model_ft = BRAUnet(img_size=256, in_chans=3, num_classes=1, n_win=8)
    model_ft = BRAUnet(img_size=224, in_chans=3, num_classes=1, n_win=7)

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
        # img = cv2.imread(f'/home/cqut/Data/medical_seg_data/ISIC2018_jpg/images/{image_id}')
        img = cv2.imread(f'/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/data/images/{image_id}')
        # mask = cv2.imread(f'/tmp/pycharm_project_794/BRAU-Netplusplus-master/isic_cvc_train_test/data/masks/{image_id}')
        # mask_dir = '/tmp/pycharm_project_794/BRAU-Netplusplus-master/isic_cvc_train_test/data/masks'
        # img = cv2.resize(img, (256, 256))
        img = cv2.resize(img, (224, 224))
        # mask = cv2.resize(mask, (256, 256))
        img_id = list(image_id.split('.'))[0]
        cv2.imwrite(f'debug/{img_id}.png', img)
        # cv2.imwrite(f'debug/{img_id}_gt.png',mask)
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        hd95_score_list = []
        for img, mask, img_id in test_dataset:
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()

            img_id = list(img_id.split('.'))[0]

            # 加载原始图像
            orig_img_path = f'/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/data/images/{img_id}.png'
            orig_img = cv2.imread(orig_img_path)
            if orig_img is None:
                print(f"Error: Could not load original image from {orig_img_path}.")
                continue

            # 将 mask 转换为二进制格式
            mask_binary = (mask.cpu().numpy()[0][0] >= 0.5).astype(np.uint8) * 255

            torch.cuda.synchronize()  # 强制同步 CUDA 操作，以确保时间测量准确
            start = time.time()
            pred = model(img)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end - start)

            pred = torch.sigmoid(pred)  # 将值压缩到 [0, 1] 区间内，通常用于二分类问题。
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred_draw = pred.clone().detach()

            if args.debug:
                # Convert prediction to binary mask for contour detection.
                pred_binary = (pred_draw.cpu().numpy()[0][0] >= 0.5).astype(np.uint8) * 255

                # Ensure the masks and original image have the same dimensions
                h, w = orig_img.shape[:2]
                mask_binary_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_binary_resized = cv2.resize(pred_binary, (w, h), interpolation=cv2.INTER_NEAREST)

                # Find and draw contours for ground truth (green color).
                contours_gt, _ = cv2.findContours(mask_binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_img, contours_gt, -1, (0, 190, 0), 1)  # Green color.

                # Find and draw contours for prediction (blue color).
                contours_pred, _ = cv2.findContours(pred_binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_img, contours_pred, -1, (255, 0, 0), 1)  # Blue color.

                # Save or show the resulting image with contours.
                cv2.imwrite(f'debug/{img_id}_contours.png', orig_img)

                # 如果需要保存预测结果图像
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                img_numpy[img_numpy == 1] = 255
                cv2.imwrite(f'debug/{img_id}_pred.png', img_numpy)

                # 保存真实标签图像
                mask_numpy = mask.cpu().numpy()[0][0]
                mask_numpy[mask_numpy == 1] = 255
                cv2.imwrite(f'debug/{img_id}_gt.png', mask_numpy)

            iouscore = iou_eval(pred, mask)
            dicescore = dice_eval(pred, mask)
            # *********************************************************
            # dice_score_func = dice_coefficient(pred, mask)
            # dice_score.append(dice_score_func.cpu().detach().numpy())

            hd95_score = hausdorff_distance_95(pred, mask)
            if hd95_score is not None:
                hd95_score_list.append(hd95_score)
            # *********************************************************
            pred = pred.view(-1)
            mask = mask.view(-1)
            # accscore = acc_eval(pred.cpu(),mask.cpu())
            accscore = acc_eval(pred.cpu(), mask.cpu().long())
            # prescore = pre_eval(pred.cpu(),mask.cpu())
            prescore = pre_eval(pred.cpu(), mask.cpu().long())
            # recallscore = recall_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(), mask.cpu().long())
            # f1score = f1_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(), mask.cpu().long())
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
        log_file.write('mean HD95: {:.4f} {:.4f}\n'.format(np.mean(hd95_score_list), np.std(hd95_score_list)))
        log_file.write('mean IoU: {:.4f} {:.4f}\n'.format(np.mean(iou_score), np.std(iou_score)))
        log_file.write('mean dice: {:.4f} {:.4f}\n'.format(np.mean(dice_score), np.std(dice_score)))
        log_file.write('mean accuracy: {:.4f} {:.4f}\n'.format(np.mean(acc_score), np.std(acc_score)))
        log_file.write('mean precision: {:.4f} {:.4f}\n'.format(np.mean(pre_score), np.std(pre_score)))
        log_file.write('mean recall: {:.4f} {:.4f}\n'.format(np.mean(recall_score), np.std(recall_score)))
        log_file.write('mean F1-score: {:.4f} {:.4f}\n'.format(np.mean(f1_score), np.std(f1_score)))
    # ***********************************************************************************
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('FPS: {:.2f}'.format(1.0 / (sum(time_cost) / len(time_cost))))
    # *****************************************************************************************
    print('mean HD95:', round(np.mean(hd95_score_list), 4), round(np.std(hd95_score_list), 4))
    # *****************************************************************************************
    print('mean IoU:', round(np.mean(iou_score), 4), round(np.std(iou_score), 4))
    print('mean dice:', round(np.mean(dice_score), 4), round(np.std(dice_score), 4))
    print('mean accuracy:', round(np.mean(acc_score), 4), round(np.std(acc_score), 4))
    print('mean precsion:', round(np.mean(pre_score), 4), round(np.std(pre_score), 4))
    print('mean recall:', round(np.mean(recall_score), 4), round(np.std(recall_score), 4))
    print('mean F1-score:', round(np.mean(f1_score), 4), round(np.std(f1_score), 4))
