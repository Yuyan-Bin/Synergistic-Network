# Bidirectional Mamba and Routing-attention Synergistic Network for Efficient Medical Image Segmentation
- Based on U-shaped architecture for medical image segmentation faces three fundamental challenges limiting clinical deployment: (1)  architectural homogeneity problems where uniform mechanism deployment overlooks distinct encoder-decoder requirements, (2) skip connection feature fusion limitations with insufficient spatial-channel attention integration, (3)  computational efficiency versus performance trade-offs. We propose a heterogeneous U-shaped architecture that strategically deploys specialized mechanisms based on component-specific functional requirements. Our approach utilizes Vision Mamba with Non-causal State Space Duality (VSSD) in encoder/bottleneck for efficient global context extraction, Bi-Level Routing Attention (BRA) in decoder for adaptive detail recovery, and introduces Spatial-Channel Synergistic Attention (SCSA) in skip connections to optimize multi-scale feature integration with only 0.1M additional parameters. Extensive experiments across four diverse datasets demonstrate exceptional performance: for example on Synapse dataset, our model achieves 84\% Dice Similarity Coefficient and 13.76mm Hausdorff Distance with only 23.43M parameters.

## 1. Synapse data and Model weights
- Get Synapse data, model weights of our model on the Synapse dataset and biformer_base_best.pth. ( https://drive.google.com/file/d/15Twkxj1Rv9J8YLCvXVBtRDPn8XAUVNWK/view?usp=sharing). I hope this will help you to reproduce the results.
- We used a dataset from the MICCAI 2015 Multi-Site CT Annotation Challenge, including 30 abdominal CT scans and 3,779 axial abdominal clinical CT images. We divided the dataset into two parts: the training set and the test set. The training set contains 18 samples with a total of 2,212 axial slices. We use these slices to train the model to learn the features of different organs. In contrast, the test set consists of the remaining 12 samples, designed to provide an independent platform for objectively evaluating the model performance.
## 2. Environment
- Please prepare an environment with python=3.10, and then use the command "pip install -r requirements.txt" for the dependencies.
## 3. Synapse Train/Test
- Train
```bash
python train.py --dataset Syanpse --root_path your DATA_DIR --max_epochs 300 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.001 --batch_size 24
```
- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 300 --base_lr 0.001 --img_size 224 --batch_size 24
```
## 4. Other databases
For training and testing on other datasets, please refer to the corresponding README.txt.
- ISIC-2018 Challenge Dataset (https://challenge.isic-archive.com/data/#2018)
The dataset consists of 2,594 dermoscopic images with real segmentation annotations. We used a five-fold cross-validation method based on the validation results and selected the best-performing model for the final prediction analysis.
- CVC-ClinicDB (https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0)
It served as the training dataset for the MICCAI 2015 Automated Polyp Detection Challenge. This dataset contains 612 images divided into three subsets: 490 for model training, 61 for performance validation, and 61 for final accuracy testing. The random assignment ensures unbiased evaluation of the model's performance.
- Rectal Cancer Segmentation Dataset (https://github.com/zf617707527/Unet1/tree/master/data1)
This study utilized the dataset from Problem B of the 2019 (7th) Teddy Cup Data Mining Challenge. The dataset contains Computed Tomography (CT) imaging data from 107 rectal cancer patients. Each patient case includes multiple slice images. We strictly excluded all negative samples to ensure data completeness and usability. This selection process guarantees that each retained CT image has detailed annotation information. After screening, the final dataset contains 860 CT images. We allocated 86 images for final testing. The remaining images use a nine-fold cross-validation method: 688 for training and 86 for validation. This method ensures both model accuracy and generalization ability.

## References
* [BiFormer](https://github.com/rayleizhu/BiFormer)
* [BRAU-Net++](https://github.com/Caipengzhou/BRAU-Netplusplus)
* [VSSD](https://github.com/YuHengsss/VSSD)
* [SCCA](https://github.com/HZAI-ZJNU/SCSA)
