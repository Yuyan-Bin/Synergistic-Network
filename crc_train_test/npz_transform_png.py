import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def npz_to_png(source_directory, output_directory):
    """
    Convert all .npz files in the specified directory to .png files and save them separately in the imagesand masksfolders.
    :param source_directory: The source directory containing .npz files  
    :param output_directory: The output directory path
    """
    images_dir = os.path.join(output_directory, 'images')
    masks_dir = os.path.join(output_directory, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(source_directory) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {source_directory}")

    for npz_file in npz_files:
        file_path = os.path.join(source_directory, npz_file)
        data = np.load(file_path)
        image = data['image']  
        label = data['label']  

        if len(image.shape) > 2:
            image = image.squeeze()
        if len(label.shape) > 2:
            label = label.squeeze()

        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        label = label.astype(np.uint8) * 255

        # Save as a .png file.
        image_filename = os.path.splitext(npz_file)[0] + '.png'
        label_filename = os.path.splitext(npz_file)[0] + '.png'

        Image.fromarray(image).save(os.path.join(images_dir, image_filename))
        Image.fromarray(label).save(os.path.join(masks_dir, label_filename))

        print(f"Saved {image_filename} to {images_dir}")
        print(f"Saved {label_filename} to {masks_dir}")

source_directory = '/tmp/pycharm_project_794/BRAU-Netplusplus-master/CRC_no/all_CRC'
output_directory = '/tmp/pycharm_project_794/BRAU-Netplusplus-master/crc_train_test/data'

npz_to_png(source_directory, output_directory)
