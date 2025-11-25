# Direction-Aware Convolution for Airway Tubular Feature Enhancement Network
## News!!
Our Paper is accepted by MedIA.
## Abstract
Manual annotation of airway regions in computed tomography images is a time-consuming and expertise dependent task. Automatic airway segmentation is therefore a prerequisite for enabling rapid bronchoscopic navigation and the clinical deployment of bronchoscopic robotic systems. Although convolutional neural network methods have gained considerable attention in airway segmentation, the unique tree-like structure of airways poses challenges for conventional and deformable convolutions, which often fail to focus on fine airway structures, leading to missed segments and discontinuities. To address this issue, this study proposes a novel tubular feature extraction network, named TfeNet. TfeNet introduces a novel direction-aware convolution operator that adapts the geometry of linear convolution kernels through spatial rotation transformations, enabling it to dy namically align with the tubular structures of airways and effectively enhance feature extraction. Furthermore, a tubular feature fusion module (TFFM) is designed based on asymmetric convolution and residual connection strategies, effectively capturing the features of airway tubules from different directions. Extensive experiments conducted on one public dataset and two datasets used in airway segmentation challenges demonstrate the effectiveness of TfeNet. Specifically, our method achieves a comprehensive lead in both accuracy and continuity on the BAS dataset, attains the highest mean score of 94.95% on the ATM22 dataset by balancing accuracy and continuity, and demonstrates superior leakage control and precision on the challenging AIIB23 dataset.

## Installation
```bash
conda create --name TfeNet python==3.8
conda activate TfeNet
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Install DAConv/DSConv
Please enter the DAConv and DSConv folders respectively to execute the installation (Make sure you have CUDA Toolkit installed):

```bash
python setup.py install
```

For more information about DAConv and DSConv , please refer to the following paper.

- Qi, Y., He, Y., Qi, X., Zhang, Y., & Yang, G. (2023). Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, 6047-6056. URL: [Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/10378018/)
- Wu, Q., Wang, Y., Zhang, Q. Direction-Aware convolution for airway tubular feature enhancement network, 2025. [URL](https://doi.org/10.1016/j.media.2025.103882)

## Datasets

- **BAS** [dataset label](https://github.com/haozheng-sjtu/3d-airway-segmentation/blob/main/BAS.rar)
  BAS dataset image info please refer to paper : Zheng, H., Qin, Y., Gu, Y., Xie, F., Yang, J., Sun, J., & Yang, G. (2020). Alleviating Class-Wise Gradient Imbalance for Pulmonary Airway Segmentation. *IEEE Transactions on Medical Imaging, 40*, 2452-2462.

- **ATM22** [challenge website](https://atm22.grand-challenge.org/)

- **AIIB23** [challenge website](https://codalab.lisn.upsaclay.fr/competitions/13238)

  **Note:** The ATM22 and AIIB23 datasets need to be registered on the challenge website and an application submitted to organizers with official email address. The validation set can be submitted online to obtain the evaluation results, while the test set needs to be submitted to the official party. The test is conducted through Docker and the evaluation feedback is combined. For more information, please visit the challenge website.


## Predict
Our trained model can be downloaded from [here](https://drive.google.com/file/d/1DEKyAMhV90AL80qoy2QgDxmO0F1cXx1r/view?usp=drive_link)


Unzip the downloaded file and place it in the ./checkpoint directory like:

```
checkpoint/
├── AIIB23
│   ├── TfeNet_checkpoint.ckpt
│   └── TfeNetSmall_checkpoint.ckpt
├── ATM22
│   ├── TfeNet_checkpoint.ckpt
│   └── TfeNetSmall_checkpoint.ckpt
└── BAS
    ├── TfeNet_checkpoint.ckpt
    └── TfeNetSmall_checkpoint.ckpt
```

TfeNet_checkpoint.ckpt and TfeNetSmall_checkpoint.ckpt is the trained model of TfeNet and TfeNetSmall respectively. 

The prediction process is divided into three steps (all case are in .nii.gz or .nii format) :
1. Perform the prediction to predict the whole airway and the small airway respectively. (You can modify the weights used in evaluation.py)
2. Combined the whole airway and small airway. (concat.py)
3. The Combined airway is post-processed to obtain the largest connected component. (postprocessing.py)

run the following command:

```bash
predict.sh # The default is the trained weights obtained through ATM22.
```

**Note:** Must set your input folder before prediction in evaluation.py(data_path = "/your/inputs"), Finished prediction results will be stored in the ./predict_result/outputs folder.

## train

The data structure we expect is as follows：

```
./BAS/
├── image
│   ├── test
│   ├── train
│   └── val
├── image_clean
│   ├── test
│   ├── train
│   └── val
├── label
│   ├── test
│   ├── train
│   └── val
├── label_clean
│   ├── test
│   ├── train
│   └── val
├── LIB_weight
│   └── train
├── LIB_weight_small
│   └── train
├── lungmask
│   ├── test
│   ├── train
│   └── val
├── lungmask_clean
│   ├── test
│   ├── train
│   └── val
├── smallairway
│   └── train
└── smallairway_clean
    ├── test
    ├── train
    └── val
```

The folders with "_clean" in their names indicate the data that has been preprocessed.

dataset prepare as follow:

1. extract lung mask
    
    ```bash
    pip install lungmask # more details please refer to https://github.com/JoHof/lungmask
    python extra_lungmask.py # please modify the path of your input and save folder.
    ```
    
2. extract small airway from lungmask and label
    
    ```bash
    python extra_smallairway.py # please modify the path of your input and save folder.
    ```
    
4. Data preprocessing involves region of interest cropping and HU adjustment. ( preprocessing.py )

5. the LIB weights for generating the entire airway and the small airways respectively. (LIB_weight.py)

6. change your data path in config of TfeNet.py ('dataset_path': r'/path/to/BAS')

7. Start training by running train.sh, see options.py for more details on how to set the training parameters.


## Evaluation for train phase
If you want to evaluate the results of the training (e.g. IOU, Precision, DSC, Sensitivity, TD, BD), you can use evaluation_meters.py.

## Citation

If you find our code or paper useful, please cite as
```latex
@article{WU2026103882,
title = {Direction-Aware convolution for airway tubular feature enhancement network},
journal = {Medical Image Analysis},
volume = {108},
pages = {103882},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103882},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525004281},
author = {Qibiao Wu and Yagang Wang and Qian Zhang},
}
```





