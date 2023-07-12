# LIGHTWEIGHT CNN FOR CROSS-VIEW GEO-LOCALIZATION USING AERIAL IMAGE

Implementation of the IGARSS2023 paper [LIGHTWEIGHT CNN FOR CROSS-VIEW GEO-LOCALIZATION USING AERIAL IMAGE]()

<img src="image/model.png" alt="model">

## Abstract

In the field of remote sensing, there has been a significant amount of research focused on linking different domains such as multi-resolution, multi-spectral, and multi-sensor imagery. One task that involves such multimodal data is cross-view geo-localization, which aims to identify the location of ground query images by matching them with aerial images in a database that is tagged with GPS information. Our study presents a novel lightweight convolutional neural network architecture that achieves comparable performance to a transformer-based model on a city-scale dataset while reducing the number of parameters without using any data augmentation and transformation. Furthermore, our experimental findings indicate that the presence or absence of the fully-connected layer, which is used for generating attention maps, has minimal influence on the model’s accuracy.

## ENVIRONMENT
Python 3.8.12 <br>
pytorch-lightning        1.8.4.post0 <br>
torch                    2.0.1 <br>
torchvision              0.15.2 <br>

## DATASETS

We use two datasets called CVUSA and CVACT in our experiments. You can access the CVUSA dataset by clicking on this link: [CVUSA dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/). Similarly, you can reach the CVACT dataset by clicking on this link: [CVACT dataset](https://github.com/Liumouliu/OriCNN).

## PRE-TRAINED MODEL

CVUSA pre-trained model: 
CVACT pre-trained model: 

## TRAIN
```
bash train_cvusa.sh

bash train_cvact.sh
```


## TEST

```
```

## Cite
If you use our implementation, please cite:
```
@InProceedings{,
    author    = {},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```
