# Modeling Mask Uncertainty in Hyperspectral Image Reconstruction


This repository contains the source code and pre-trained models for the paper [**Modeling mask uncertainty in hyperspectral image reconstruction**](https://arxiv.org/pdf/2112.15362.pdf) by Jiamian wang, [Yulun Zhang](http://yulunzhang.com/), [Xin Yuan](https://xygroup6.github.io/xygroup/), [Ziyi Meng](https://github.com/mengziyi64), and [Zhiqiang Tao](https://ztao.cc/).

## Updates
More pre-trained models of compared methods will come soon!:rocket:

## Introduction
Recently, hyperspectral imaging (HSI) has attracted increasing research attention, especially for the ones based on a coded aperture snapshot spectral imaging (CASSI) system. Existing deep HSI reconstruction models are generally trained on paired data to retrieve original signals upon 2D compressed measurements given by a particular optical hardware mask in CASSI, during which the mask largely impacts the reconstruction performance and could work as a “model hyperparameter” governing on data augmentations. This mask-specific training style will lead to a hardware miscalibration issue, which sets up barriers to deploying deep HSI models among different hardware and noisy environments. To address this challenge, we introduce mask uncertainty for HSI with a complete variational Bayesian learning treatment and explicitly model it through a mask decomposition inspired by real hardware. Specifically, we propose a novel Graph-based Self-Tuning (GST) network to reason uncertainties adapting to varying spatial structures of masks among dif- ferent hardware. Moreover, we develop a bilevel optimization framework to balance HSI reconstruction and uncertainty estimation, accounting for the hyperparameter property of masks. Extensive experimental results validate the effectiveness (over 33/30 dB) of the proposed method under two miscalibration scenarios and demonstrate a highly competitive performance compared with the state-of-the-art well-calibrated methods.


![RDN](/figure/framework.png)
Figure 1. Illustration of modeling mask uncertainty with the proposed Graph-based Self-Tuning (GST) network


## Performance

Table 1. PSNR(dB)/SSIM by different methods on 10 simulation scenes under the many-to-many hardware miscalibration. All the methods are trained with a mask set and tested by random unseen masks. TSA-Net, GSM, and SRN are obtained with a mask ensemble strategy. We report mean/std among 100 testing trials.
![RDN](/figure/M2M_tab.png)


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

Ten 256x256x28 and ten 256x256x24 benchmark testing data are provided. 
Testing trials can be determined by specify `trial_num`

For the 256x256x28 data type (employed in the main paper), pre-trained models of traditional/miscalibration scenarios are provided. Please specify `last_train` as 623 for `mode` `one_to_one` and `one_to_many`. Specify `last_train` as 661 for `many_to_many`. For example, run

```
python test.py --mode many_to_many --trial_num 100 --last_train 661 --test_data_type 28chl --model_type GST --test_path ./Data/testing/28chl/ --inter_channels 28 --spatial_scale 4 --noise_act softplus
```
```
python test.py --mode one_to_many --trial_num 100 --last_train 623 --test_data_type 28chl --model_type GST --test_path ./Data/testing/28chl/ --inter_channels 28 --spatial_scale 4 --noise_act softplus
```
```
python test.py --mode one_to_one  --last_train 623 --test_data_type 28chl --model_type GST --test_path ./Data/testing/28chl/ --inter_channels 28 --spatial_scale 4 --noise_act softplus
```

For the 256x256x24 data type, pre-trained model of `many_to_many` miscalibration scenario (primary concern) is provided.  For example, run

```
python test.py --test_path ./Data/testing/24chl/test.mat --mode many_to_many --trial_num 100 --last_train 654 --test_data_type 24chl --model_type ST --noise_act softplus 
```



## Structure of directories

| directory  | description  |
| :--------: | :----------- | 
| `Data` | Ten 256x256x28 testing data and ten 256x256x24 testing data | 
| `test`    | testing script |
| `utils`   | utility functions|
| `network`    | GST network and simplified version (ST) |
| `ssim_torch`    | function for computing SSIM |
| `model`      | pre-trained models for both 28-channel and 24-channel HSI data |
| `train`| training script |


## Citation

If you find the code helpful in your resarch, please kindly cite the following papers.
```
@article{wang2021calibrated,
  title={Calibrated Hyperspectral Image Reconstruction via Graph-based Self-Tuning Network},
  author={Wang, Jiamian and Zhang, Yulun and Yuan, Xin and Meng, Ziyi and Tao, Zhiqiang},
  journal={arXiv preprint arXiv:2112.15362},
  year={2021}
}

@article{wang2021new,
  title={A new backbone for hyperspectral image reconstruction},
  author={Wang, Jiamian and Zhang, Yulun and Yuan, Xin and Fu, Yun and Tao, Zhiqiang},
  journal={arXiv preprint arXiv:2108.07739},
  year={2021}
}
```

## Contact

If you have any questions, please contact Jiamian Wang (jiamiansc@gmail.com).



## Acknowledgements

We refer to the [TSA-Net](https://github.com/mengziyi64/TSA-Net) when we develop this code.  Great thanks to them!
