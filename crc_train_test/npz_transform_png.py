import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def npz_to_png(source_directory, output_directory):
    """
    将指定目录下的所有 .npz 文件转换为 .png 文件，并分别保存到 images 和 masks 文件夹中。

    :param source_directory: 包含 .npz 文件的源目录
    :param output_directory: 输出目录路径
    """
    # 创建目标文件夹
    images_dir = os.path.join(output_directory, 'images')
    masks_dir = os.path.join(output_directory, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 获取所有 .npz 文件
    npz_files = [f for f in os.listdir(source_directory) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {source_directory}")

    # 遍历每个 .npz 文件
    for npz_file in npz_files:
        file_path = os.path.join(source_directory, npz_file)
        data = np.load(file_path)
        image = data['image']  # 加载图像数据
        label = data['label']  # 加载标签数据

        # 确保图像是二维数组
        if len(image.shape) > 2:
            image = image.squeeze()
        if len(label.shape) > 2:
            label = label.squeeze()

        # 将图像数据转换为 8-bit 格式（0-255）
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)

        # 将标签数据转换为 8-bit 格式（0-255）
        label = label.astype(np.uint8) * 255

        # 保存为 .png 文件
        image_filename = os.path.splitext(npz_file)[0] + '.png'
        label_filename = os.path.splitext(npz_file)[0] + '.png'

        Image.fromarray(image).save(os.path.join(images_dir, image_filename))
        Image.fromarray(label).save(os.path.join(masks_dir, label_filename))

        print(f"Saved {image_filename} to {images_dir}")
        print(f"Saved {label_filename} to {masks_dir}")

# 指定路径
source_directory = '/tmp/pycharm_project_794/BRAU-Netplusplus-master/CRC_no/all_CRC'
output_directory = '/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/data'

# 调用函数
npz_to_png(source_directory, output_directory)