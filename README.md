# Modeling Mask Uncertainty in Hyperspectral Image Reconstruction


This repository contains the source code and pre-trained models for the paper **Modeling mask uncertainty in hyperspectral image reconstruction**.

## Introduction
Recently, hyperspectral imaging (HSI) has attracted increasing research attention, especially for the ones based on a coded aperture snapshot spectral imaging (CASSI) system. Existing deep HSI reconstruction models are generally trained on paired data to retrieve original signals upon 2D compressed measurements given by a particular optical hardware mask in CASSI, during which the mask largely impacts the reconstruction performance and could work as a “model hyperparameter” governing on data augmentations. This mask-specific training style will lead to a hardware miscalibration issue, which sets up barriers to deploying deep HSI models among different hardware and noisy environments. To address this challenge, we introduce mask uncertainty for HSI with a complete variational Bayesian learning treatment and explicitly model it through a mask decomposition inspired by real hardware. Specifically, we propose a novel Graph-based Self-Tuning (GST) network to reason uncertainties adapting to varying spatial structures of masks among dif- ferent hardware. Moreover, we develop a bilevel optimization framework to balance HSI reconstruction and uncertainty estimation, accounting for the hyperparameter property of masks. Extensive experimental results validate the effectiveness (over 33/30 dB) of the proposed method under two miscalibration scenarios and demonstrate a highly competitive performance compared with the state-of-the-art well-calibrated methods.

![RDB](/figure/bilevel.png)
Figure 1. Illustration of the proposed bilevel optimization framework
![RDN](/figure/framework.png)
Figure 2. Illustration of modeling mask uncertainty with the proposed Graph-based Self-Tuning (GST) network

## Requirements

* Python 3.7.10
* Pytorch 1.9.1
* Numpy 1.21.2
* Scipy 1.7.1

## Train

Run

```
python train.py
```


## Test

28-channels dataset

24-channels dataset

Ten simulation testing HSI (256x256x28) are provided. Testing trials can be determined by specify `trial_num`

To test a pre-trained model under miscalibration many-to-many, specify `mode` as many_to_many, `last_train` as 661. 


To test a pre-trained model under miscalibration one-to-many, specify `mode` as one_to_many, `last_train` as 662. 


To test a pre-trained model under traditional setting one-to-one, specify `mode` as one_to_one, `last_train` as 662. 

Run

```
python test.py
```



## Structure of directories

| directory  | description  |
| :--------: | :----------- | 
| `Data` | Ten simulation testing HSIs and two real masks for testing (256x256 and 660x660) | 
| `test`    | testing script |
| `utils`   | utility functions|
| `tools`    | model components |
| `ssim_torch`    | function for computing SSIM |
| `many_to_many`      | model structure and model checkpoint for miscalibration (many-to-many) |
| `one_to_many`      | model structure and model checkpoint for miscalibration (one-to-many) |
| `one_to_one`      | model structure and model checkpoint for traditional setting (one-to-one) |


