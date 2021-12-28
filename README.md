# mask-uncertainty-in-HSI


This anonymized repository contains the testing code and pre-trained models for the paper **Calibrated Hyperspectral Image Reconstruction via Graph-based Self-Tuning Network**, which is submitted at CVPR 2022 with paper ID: **4875**.

## Requirements

* Python 3.7.10
* Pytorch 1.9.1
* Numpy 1.21.2
* Scipy 1.7.1

## Test

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



# mask_uncertainty_spectral_SCI
