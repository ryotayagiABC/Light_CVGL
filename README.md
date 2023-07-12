# LIGHTWEIGHT CNN FOR CROSS-VIEW GEO-LOCALIZATION USING AERIAL IMAGE

Implementation of the IGARSS2023 paper [LIGHTWEIGHT CNN FOR CROSS-VIEW GEO-LOCALIZATION USING AERIAL IMAGE]()

<img src="image/model.png" alt="model">

## Abstract

In the field of remote sensing, there has been a significant amount of research focused on linking different domains such as multi-resolution, multi-spectral, and multi-sensor imagery. One task that involves such multimodal data is cross-view geo-localization, which aims to identify the location of ground query images by matching them with aerial images in a database that is tagged with GPS information. Our study presents a novel lightweight convolutional neural network architecture that achieves comparable performance to a transformer-based model on a city-scale dataset while reducing the number of parameters without using any data augmentation and transformation. Furthermore, our experimental findings indicate that the presence or absence of the fully-connected layer, which is used for generating attention maps, has minimal influence on the model’s accuracy.

## ENVIRONMENT
```
Python 3.8.12
```
Main Library
```
pytorch-lightning        1.8.4.post0 
torch                    2.0.1 
torchvision              0.15.2 
opencv                    4.4.0 
```
Please install any other necessary items.

## DATASETS

We use two datasets called CVUSA and CVACT in our experiments. You can access the CVUSA dataset by clicking on this link: [CVUSA dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/). Similarly, you can reach the CVACT dataset by clicking on this link: [CVACT dataset](https://github.com/Liumouliu/OriCNN).

Once you have downloaded the datasets, we kindly suggest organizing them in the following directory structure.
```
CVACT
├── ACT_data.mat
├── grd
└── sat

CVUSA
├── grd
├── sat
└── splits
    ├── train-19zl.csv
    └── val-19zl.csv
```

## PRE-TRAINED MODEL

CVUSA pre-trained model: [here](https://drive.google.com/file/d/1np4jkdDCddwjVZxBMJPi9WYAgDqEhu-K/view?usp=drive_link)
CVACT pre-trained model: [here](https://drive.google.com/file/d/1QrsggSbTczECG39S8EfCR9XsmdjEJ9e8/view?usp=drive_link)

## TRAIN
For CVUSA, please run the script below.
```
bash train_cvusa.sh
```
For CVACT, please run the script below.
```
bash train_cvact.sh
```

Our code is implemented using pytorch-lightning so we can train using multiple GPUs.　To use multiple GPUs, rewrite the shell script as follows.
```
--gpu 0,1
```

## TEST
Specify the checkpoint by changing the shell script below.
```
--checkpoint "./weight/checkpoint.pth"
```
During the testing phase, you should select only one GPU.

For CVUSA, please run the script below.
```
bash test_cvusa.sh
```
For CVACT, please run the script below.
```
bash test_cvact.sh
```

## Cite
If you use our implementation, please cite:
```
Coming soon.
```
