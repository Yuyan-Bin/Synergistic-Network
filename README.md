# Bidirectional Mamba and Routing-attention Synergistic Network for Efficient Medical Image Segmentation
## Description
This repository contains the official implementation for **"Bidirectional Mamba and Routing-attention Synergistic Network for Efficient Medical Image Segmentation"**.
**HeteroMedSeg** overcomes the limitations of homogeneous U-Net architectures by introducing a **component-specialized design** for medical image segmentation. Instead of uniform mechanisms, our model strategically employs:
-   A **Vision Mamba (VSSD)** encoder for efficient long-range context modeling.
-   A **Bi-Level Routing Attention (BRA)** decoder for precise detail recovery.
-   A lightweight **Spatial-Channel Synergistic Attention (SCSA)** module (<0.01M params) in skip connections for enhanced feature fusion.

This novel synergy achieves a superior accuracy-efficiency balance, exemplified by a **84.0% DSC** and **13.76mm HD** on the Synapse dataset with only **23.43M parameters**.

## 1. Model structure.
- The overview of our model.
  ![image](https://github.com/Yuyan-Bin/Synergistic-Network/blob/master/model.png)
## 2. Environment
- Please prepare an environment with python=3.10, and then use the command "pip install -r requirements.txt" for the dependencies.
## 3. Dataset information
- Synapse Multi-Organ Segmentation Dataset (official website: https://www.synapse.org/Synapse:syn3193805/wiki/)
  - Get Synapse data, model weights of our model on the Synapse dataset and biformer_base_best.pth. ( https://drive.google.com/file/d/15Twkxj1Rv9J8YLCvXVBtRDPn8XAUVNWK/view?usp=sharing). I hope this will help you to reproduce the results.
  - We used a dataset from the MICCAI 2015 Multi-Site CT Annotation Challenge, including 30 abdominal CT scans and 3,779 axial abdominal clinical CT images. We divided the dataset into two parts: the training set and the test set. The training set contains 18 samples with a total of 2,212 axial slices. We use these slices to train the model to learn the features of different organs. In contrast, the test set consists of the remaining 12 samples, designed to provide an independent platform for objectively evaluating the model performance.
- ISIC-2018 Challenge Dataset (official website: https://challenge.isic-archive.com/data/#2018)
  - The dataset consists of 2,594 dermoscopic images with real segmentation annotations. We used a five-fold cross-validation method based on the validation results and selected the best-performing model for the final prediction analysis.
- CVC-ClinicDB (official website: https://polyp.grand-challenge.org/CVCClinicDB/)
  - It served as the training dataset for the MICCAI 2015 Automated Polyp Detection Challenge. This dataset contains 612 images divided into three subsets: 490 for model training, 61 for performance validation, and 61 for final accuracy testing. The random assignment ensures unbiased evaluation of the model's performance.
- Rectal Cancer Segmentation Dataset (PaddlePaddle AI Studio: https://github.com/zf617707527/Unet1/tree/master/data1)
  - This study utilized the dataset from Problem B of the 2019 (7th) Teddy Cup Data Mining Challenge. The dataset contains Computed Tomography (CT) imaging data from 107 rectal cancer patients. Each patient case includes multiple slice images. We strictly excluded all negative samples to ensure data completeness and usability. This selection process guarantees that each retained CT image has detailed annotation information. After screening, the final dataset contains 860 CT images. We allocated 86 images for final testing. The remaining images use a nine-fold cross-validation method: 688 for training and 86 for validation. This method ensures both model accuracy and generalization ability.
## 4. Code Information
Extract the files from https://drive.google.com/file/d/15Twkxj1Rv9J8YLCvXVBtRDPn8XAUVNWK/view?usp=sharing, place the biformer_base_best.pth in the /synapse_train_test/pretrained_ckpt folder, download the corresponding dataset, and place it according to the following instructions. The synapse_84.00.pth file is the best model file from the Synapse dataset and should be placed in the /synapse_train_test/save_models folder.
- To apply the model on Synapse dataset, the data tree should be constructed as:
``` 
    ├── data
          ├── Synapse
                ├── test_vol_h5
                      ├── image_1.npz.h5
                      ├── image_2.npz.h5
                      ├── image_n.npz.h5
                ├── train_npz
                      ├── image_1.npz
                      ├── image_2.npz
                      ├── image_n.npz
```
- To apply the model on other datasets, the data tree should be constructed as:
``` 
    ├── data
          ├── images
                ├── image_1.{png,tif}
                ├── image_2.{png,tif}
                ├── image_n.{png,tif}
          ├── masks
                ├── image_1.{png,tif}
                ├── image_2.{png,tif}
                ├── image_n.{png,tif}
```

## 5. Installation
### Step 1: Clone the repository:
```bash
git clone https://github.com/Yuyan-Bin/Synergistic-Network.git
cd Synergistic-Network
```
### Step 2: Environment Setup:
#### Create and activate a new conda environment
```bash
conda create -n Synergistic-Network
conda activate Synergistic-Network
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```


## 6. Quick Start
### Synapse Train/Test
- Train
```bash
python train.py --dataset Syanpse --root_path your DATA_DIR --max_epochs 300 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.001 --batch_size 24
```
- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 300 --base_lr 0.001 --img_size 224 --batch_size 24
```
### ISIC-2018 Train/Test
- CSV generation 
```
python data_split_csv.py --dataset your/data/path --size 0.9 
```
- Train
```
python train.py --dataset your/data/path --csvfile your/csv/path --loss dice --batch 42 --lr 0.0005 --epoch 200 
```
- Test
```
python test.py --dataset your/data/path --csvfile your/csv/path --model save_models/epoch_last.pth --debug True
```
### CVC-ClinicDB and Rectal Cancer Segmentation Dataset Train/Test
- CSV generation 
```
python data_split_csv.py --dataset your/data/path --size 0.9 
```
- Train
```
python train.py --dataset your/data/path --csvfile your/csv/path --loss dice --batch 44 --lr 0.0006 --epoch 200 
```
- Test
```
python test.py --dataset your/data/path --csvfile your/csv/path --model save_models/epoch_last.pth --debug True
```
## 7. Obtain the outputs.
- After trianing, you could obtain the outputs in './save_models/'
## References
* [BiFormer](https://github.com/rayleizhu/BiFormer)
* [BRAU-Net++](https://github.com/Caipengzhou/BRAU-Netplusplus)
* [VSSD](https://github.com/YuHengsss/VSSD)
* [SCCA](https://github.com/HZAI-ZJNU/SCSA)

## License

Distributed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

> For more details about the Apache-2.0 license, please visit [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).
