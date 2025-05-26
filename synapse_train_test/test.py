import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from matplotlib.colors import ListedColormap, Normalize
# from skimage.color import rgb2srgb, rgb2adobergb, rgb2prophoto
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
pl = platform.system()
if pl != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
from datasets.dataset_synapse import Synapse_dataset
from networks.bra_unet import BRAUnet
from utils import test_single_volume

parser = argparse.ArgumentParser()

draw = 'true'

parser.add_argument(
    "--volume_path",
    type=str,
    # default="/home/cqut/Data/medical_seg_data/Synapse_zheng",
    default="/tmp/pycharm_project_794/BRAU-Netplusplus-master/synapse_train_test/data/Synapse",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--output_dir", type=str, default="./save_models", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=280, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--is_savenii", action="store_true",default=True,help="whether to save results during inference")
# ****************************************更改路**********************************************
parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
# parser.add_argument("--test_save_dir", tyype=str, default=None, help="saving prediction as nii!")
# ****************************************************************************************************************************
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)
#
# # *******************绘图******************************************************
def save_figure(fig, filename, folder=args.test_save_dir):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    absolute_filepath = os.path.abspath(filepath)  # 获取绝对路径
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Figure saved to {absolute_filepath}")  # 打印绝对路径
    # 检查文件是否真的被创建
    # 检查文件是否真的被创建
    if os.path.isfile(absolute_filepath):
        print(f"File exists and was saved successfully at {absolute_filepath}.")
    else:
        print(f"Failed to save file at {absolute_filepath}.")


def label2color(label, colormap):
    """将标签图转换为彩色图像"""
    norm = Normalize(vmin=0, vmax=colormap.N)
    return colormap(norm(label))


def plot_images(original, predicted, label, title="Image Comparison", color_space='sRGB'):
    # Convert tensors to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.squeeze().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.squeeze().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.squeeze().cpu().numpy()

    colors = [(0, 0, 0),  # Background
              (255, 0, 0),  # Class 1 蓝色改红色
              (0, 255, 0),  # Class 2 绿色 T
              (100, 0, 255),  # Class 3 红色改紫色
              (0, 255, 255),  # Class 4 青蓝色 T
              (255, 52, 255),  # Class 5 玫红 T
              (255, 255, 0),  # Class 6 黄色 T
              (255, 239, 213),  # Class 7 浅黄色替代浅蓝色
              (0, 0, 215)]  # Class 8 深蓝替代灰色

    # 将颜色从 [0, 255] 范围转换到 [0, 1] 范围
    colors_normalized = np.array(colors) / 255.0
    colormap = ListedColormap(colors_normalized)

    def create_rgba(image_colored, alpha_value=0.5, color_space='sRGB'):
        """创建带有透明度通道的RGBA图像"""
        # 创建 alpha 通道，黑色部分设置为透明
        alpha_channel = np.full(image_colored.shape[:2], alpha_value, dtype=np.float32)
        black_pixels = np.all(image_colored == [0, 0, 0], axis=-1)
        alpha_channel[black_pixels] = 0

        # 根据选定的色彩空间调整颜色
        if color_space.lower() == 'adobe rgb':
            image_colored = rgb2adobergb(image_colored)
        elif color_space.lower() == 'prophoto rgb':
            image_colored = rgb2prophoto(image_colored)
        else:  # 默认使用 sRGB
            image_colored = rgb2srgb(image_colored)

        return np.dstack((image_colored, alpha_channel))

    # 如果label是二维的，那么不需要遍历切片
    if label.ndim == 2:
        if np.any(label > 0):
            label_colored = label2color(label, colormap)
            predict_colored = label2color(predicted, colormap)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 显示原图像
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')

            # 处理预测图像
            predict_colored_rgba = create_rgba(predict_colored, color_space=color_space)
            axes[1].imshow(original, cmap='gray')
            axes[1].imshow(predict_colored_rgba)
            axes[1].set_title("Predicted")
            axes[1].axis('off')

            # 处理标签图像
            label_colored_rgba = create_rgba(label_colored, color_space=color_space)
            axes[2].imshow(original, cmap='gray')
            axes[2].imshow(label_colored_rgba)
            axes[2].set_title("Label")
            axes[2].axis('off')

            plt.suptitle(f"{title} ({color_space})")
            save_figure(fig, f"{title}_{color_space}.png")

    else:
        # 如果label是三维的，遍历每个切片
        for i in range(label.shape[0]):
            if np.any(label[i, :, :] > 0):
                original_slice = original[i, :, :] if original.ndim == 3 else original
                predicted_slice = predicted[i, :, :] if predicted.ndim == 3 else predicted
                label_slice = label[i, :, :] if label.ndim == 3 else label

                label_colored = label2color(label_slice, colormap)
                predict_colored = label2color(predicted_slice, colormap)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(original_slice, cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis('off')

                # Process and display predicted image
                predict_colored_rgba = create_rgba(predict_colored, color_space=color_space)
                axes[1].imshow(original_slice, cmap='gray')
                axes[1].imshow(predict_colored_rgba)
                axes[1].set_title("Predicted")
                axes[1].axis('off')

                # Process and display label image
                label_colored_rgba = create_rgba(label_colored, color_space=color_space)
                axes[2].imshow(original_slice, cmap='gray')
                axes[2].imshow(label_colored_rgba)
                axes[2].set_title("Label")
                axes[2].axis('off')

                plt.suptitle(f"{title} - Slice {i} ({color_space})")
                save_figure(fig, f"{title}_slice_{i}_{color_space}.png")
def inference(args, model, test_save_path=None, color_space='sRGB'):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        print(f"Batch {i_batch} image tensor shape: {sampled_batch['image'].shape}")
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        print(f"Batch {i_batch} input tensor shape: {image.shape}")
        metric_i, predicted = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        # if draw == 'true':
        #     print("draw:true;")
        #     plot_images(image, predicted, label, title=case_name, color_space=color_space)
        # 测试图片尺寸
        patch_size = [args.img_size, args.img_size]
        print(f"Patch size for testing: {patch_size}")

        metric_list += np.array(metric_i)
        logging.info(
            "idx %d case %s mean_dice %f mean_hd95 %f"
            % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
        )
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f" % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f" % (performance, mean_hd95))
    return "Testing Finished!"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # ***********************************************************************************************************
    new_directory = "/tmp/pycharm_project_794/BRAU-Netplusplus-master/synapse_train_test"

    # 尝试切换目录
    try:
        os.chdir(new_directory)
        print(f"Changed working directory to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Error: The directory '{new_directory}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied to change to directory '{new_directory}'.")
    # *********************************************************************************************************

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    net = BRAUnet(img_size=224,in_chans=3, num_classes=9, n_win=7).cuda(0)

    # snapshot = os.path.join(args.output_dir, "best_model.pth")
    snapshot = os.path.join(args.output_dir, "best_model.pth")
    # if not os.path.exists(snapshot):
    #     snapshot = snapshot.replace("best_model", "synapse_epoch_299")
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "synapse_epoch_249_bs_24_20250415_1637")
    msg = net.load_state_dict(torch.load(snapshot),strict=False)

    print("self trained swin unet", msg)
    snapshot_name = snapshot.split("/")[-1]

    log_folder = "./test_log/test_log_"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",

        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    print("args.is_savenii",args.is_savenii)
    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
