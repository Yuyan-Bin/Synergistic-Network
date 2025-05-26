import os

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
def calculate_metric_percase(pred, gt):
    # ************************************************************************************
    # 确保 pred 和 gt 的形状一致
    if pred.shape != gt.shape:

        gt = zoom(gt, (pred.shape[0] / gt.shape[0], pred.shape[1] / gt.shape[1]), order=0)
    # ************************************************************************************
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() #squeeze(0) 会移除第一个维度（批量维度）
    # print("original image.shape:",image.shape)
    # print("original label.shape:", label.shape)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        # print("标签形状1：",prediction.shape)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # ***********************************
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(slice).unsqueeze(0).float().cuda()
            # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # ***********************************
            # print("评估前维度1：", input.shape)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                # print("if 分支里面out.shape",out.shape)
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                # print("pred.shape", pred.shape)
                prediction[ind] = pred
    else:
        # *************************************************
        # print("标签形状2：",prediction.shape)
        assert len(image.shape) == 2, "Input image must be 2D if not 3D."
        x, y = image.shape[:2]  # 获取原始图像的宽度和高度
        # print("patch_size:",patch_size[0])
        # 调整图像大小以匹配 patch_size
        if x != patch_size[0] or y != patch_size[1]:
            image_resized = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        else:
            image_resized = image
        # *************************************************
        input = torch.from_numpy(image_resized).unsqueeze(
            0).unsqueeze(0).float().cuda()  # 插入两个维度  （512，512）变为 (1, 1, 512, 512)
        # print("评估前维度2：",input.shape)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            # print("else 分支里面out.shape", out.shape)
            prediction = out.cpu().detach().numpy()
            # print("prediction.shape", prediction.shape)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list, prediction