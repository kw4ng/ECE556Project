# An Efficient Multiscale Spatial Rearrangement MLP Architecture for Image Restoration

This repository is a student implementation of [An Efficient Multiscale Spatial Rearrangement MLP Architecture for Image Restoration](https://ieeexplore.ieee.org/document/10373791). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Model Implementation Files
Our model is implemented in the following files:
```
srm.py
linear_gating.py
mssr_mlp.py
mssr.py
channel_mlp.py
unet.py
```

Each file contains a module or component of the overall model, as defined by the file name. 

## Dataset Preparation
Download the GOPRO_Large dataset from [https://seungjunnah.github.io/Datasets/gopro.html](https://seungjunnah.github.io/Datasets/gopro.html). Run this command to generate image patches for both training and testing:

```
python generate_patches.py
```

The final dataset should be structured as follows:
```
Datasets/
├── train/
│   ├── GoPro/ 
│       ├── input_crops/
│       └── target_crops/
└── test/
    ├── GoPro/ 
        ├── input_crops/
        └── target_crops/
```

## Training

To train the model:

```train
python train.py
```
It will then prompt you to define the number of epochs and the batch size. The default hyperparameters are defined in the code and can be altered as needed.

## Evaluation

After training, there will be a saved .pth file with the model weights. To evaluate with the trained model:

```eval
python test.py \
    --weights [path_to_checkpoint.pth] \
    --input_dir Datasets/test/GoPro/input_crops/ \
    --target_dir Datasets/test/GoPro/target_crops/ \
    --output_dir [path_to_output_images]
```
This will run the model on all the input images, calculate the average PSNR and SSIM, and print the results on the terminal.
## Pre-trained Models
Our pretrained model is in the repository as:
```
.pth
```

## Results

Our model achieves the following performance:

| Model | Average PSNR (dB) | Average SSIM | FLOPs (G) | Params (M) |
| - | - | - | - | - | 
| Our model |     85%         |      95%       |
|Paper's model | | | | |
