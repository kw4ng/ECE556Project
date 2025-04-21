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
gopro_mssrmlp_b_e10_bs4_loss0.288616.pth
```

## Results

Our model achieves the following performance:

| Model | Average PSNR (dB) | Average SSIM | FLOPs (G) | Params (M) |
| - | - | - | - | - | 
| Our model | 28.1463 | 0.8275 | 75.74 | 11.11 |
| Paper's model | 33.23 | 0.962 | 64.06 | 15.68 |
| MAXIM (MLP-based model) | 32.86 | 0.961 | 339.2 | 22.20 |
| one more |

Our model does not achieve the performance that the authors of the paper were able to. However, our model was trained on only subset of the training dataset, a fraction of the number of epochs (10 vs 3000), and is smaller in size. Given our limited time and resources, we believe it was successful. 
