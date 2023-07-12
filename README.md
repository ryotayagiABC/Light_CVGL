# LIGHTWEIGHT CNN FOR CROSS-VIEW GEO-LOCALIZATION USING AERIAL IMAGE

Implementation of the IGARSS2023 paper [LIGHTWEIGHT CNN FOR CROSS-VIEW GEO-LOCALIZATION USING AERIAL IMAGE]()

<img src="image/model.png" alt="model">

## Abstract

In the field of remote sensing, there has been a significant amount of research focused on linking different domains such as multi-resolution, multi-spectral, and multi-sensor imagery. One task that involves such multimodal data is cross-view geo-localization, which aims to identify the location of ground query images by matching them with aerial images in a database that is tagged with GPS information. Our study presents a novel lightweight convolutional neural network architecture that achieves comparable performance to a transformer-based model on a city-scale dataset while reducing the number of parameters without using any data augmentation and transformation. Furthermore, our experimental findings indicate that the presence or absence of the fully-connected layer, which is used for generating attention maps, has minimal influence on the model’s accuracy.

## ENVIRONMENT
Python 3.8.12
pytorch-lightning        1.8.4.post0
torch                    2.0.1
torchvision              0.15.2

## PRE-TRAINED MODEL

CVUSA pre-trained model: 
CVACT pre-trained model: 

## TRAIN



## TEST



## Cite
If you use our implementation, please cite:
```
@InProceedings{Toker_2021_CVPR,
    author    = {Toker, Aysim and Zhou, Qunjie and Maximov, Maxim and Leal-Taixe, Laura},
    title     = {Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {6488-6497}
}
```
