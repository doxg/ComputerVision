<p float="left">
  <img align:"left" src = "https://miro.medium.com/max/1200/1*bBS_lYMoWhiyJf733Bghwg.jpeg" width="50%" height = "150"> 
  <img align: "right" src = "https://albumentations.ai/assets/img/custom/albumentations_card.png" width="50%" height = "150">
</p>
--------------------------------------------------------------------------------


<h1 align="center"> UNET for semantic segmentation </h1>

## Introduction
Segmentation is one of basic tasks in computer vision. In segmetation, every single pixel of image is represented as certain class, while the whole image is labeled in standard 
classsification task. The output of model is high resolution image(usually the same size as input) in which each pixel is labeled. 

** Semantic segmentation vs Instance segmentation: **

Semantic segmenation doesn't distinguish unique objects within a single category and all of them are treated as one entity.
Instance segmenation identified individual objects related to one category.

## UNET
If we take a close look at architecture of the network, it resembles character "U". The main idea of network is to make a symmetrical shape by successive layers, where there is an
upsampling operator for each pooling operator. Upsampling is done on output of contracting layers, which contains most of extracted context information. However, there might be
some local details lost during pooling. So, shortcut pathes between corresponding contracting and expansive layer are added in order to make upsampling more accurate. The output of 
each contracting layer before being pooled are sent to corresponding expansive layer. In such a creative way, the potentially lost data can be compensated.

<img src ="https://github.com/doxg/ComputerVision/blob/master/Architechture.png">

The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step the number of features channels is doubled.
Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”, "transpose conv") that halves the
number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path (shortcut path), and two 3x3 convolutions, each followed by 
a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to
the desired number of classes. In total the network has 23 convolutional layers.

### Dataset


### References:
1) https://arxiv.org/pdf/1505.04597.pdf
2) https://towardsdatascience.com/u-net-b229b32b4a71
3) https://github.com/aladdinpersson/Machine-Learning-Collection
4) https://github.com/bnsreenu/python_for_microscopists/blob/master/214_multiclass_Unet_sandstone_segm_models_ensemble.py
