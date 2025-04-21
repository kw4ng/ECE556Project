## Reference: https://github.com/vztu/maxim-pytorch/blob/main/Deblurring/generate_patches_gopro.py

##### Data preparation file for training Restormer on the GoPro Dataset ########

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
# !pip install pdb
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing


def train_files(file_):
    lr_file, hr_file = file_
    # original base filename, e.g. “0001”
    base = os.path.splitext(os.path.basename(lr_file))[0]
    # scene folder, e.g. “scene02”
    scene = os.path.basename(os.path.dirname(os.path.dirname(lr_file)))
    
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]

    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=int))
        h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=int))
        w1.append(w - patch_size)
        h1.append(h - patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                lr_patch = lr_img[i:i + patch_size, j:j + patch_size, :]
                hr_patch = hr_img[i:i + patch_size, j:j + patch_size, :]

                # now include scene and patch index in the name
                out_name = f"{scene}_{base}-{num_patch}.png"
                lr_savename = os.path.join(lr_tar, out_name)
                hr_savename = os.path.join(hr_tar, out_name)

                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)
    else:
        # single full‐image case
        out_name = f"{scene}_{base}.png"
        lr_savename = os.path.join(lr_tar, out_name)
        hr_savename = os.path.join(hr_tar, out_name)

        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)



def test_files(file_):
    lr_file, hr_file = file_
    # extract base name and scene name
    base     = os.path.splitext(os.path.basename(lr_file))[0]                     # e.g. "0001"
    scene    = os.path.basename(os.path.dirname(os.path.dirname(lr_file)))        # e.g. "scene02"
    out_name = f"{scene}_{base}.png"                                              # "scene02_0001.png"

    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)

    # center‐crop
    w, h = lr_img.shape[:2]
    i = (w - val_patch_size) // 2
    j = (h - val_patch_size) // 2
    lr_patch = lr_img[i:i+val_patch_size, j:j+val_patch_size]
    hr_patch = hr_img[i:i+val_patch_size, j:j+val_patch_size]

    # save under unique name
    lr_savename = os.path.join(lr_tar, out_name)
    hr_savename = os.path.join(hr_tar, out_name)
    cv2.imwrite(lr_savename, lr_patch)
    cv2.imwrite(hr_savename, hr_patch)



############ Prepare Training data ####################
num_cores = 10
patch_size = 256  # training on 256 * 256 patches
overlap = 128
p_max = 0

src = 'GOPRO_Large/train'  # raw dataset location
tar = 'Datasets/train/GoPro'  # cropped dataset new location

lr_tar = os.path.join(tar, 'input_crops')  # blurry cropped picture
hr_tar = os.path.join(tar, 'target_crops')  # sharp cropped corresponding picture

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

# Get images from the files, since we did not download via their python code we need to add '*' bc /input/scene../images
# instead of directly /input/images
lr_files = natsorted(glob(os.path.join(src, '*', 'blur', '*.png')))
hr_files = natsorted(glob(os.path.join(src, '*', 'sharp', '*.png')))

# Confirm Path correct?
files = [(i, j) for i, j in zip(lr_files, hr_files)]  # pair up blurry and sharp images
print(f"Found {len(lr_files)} blurry images")
print(f"Found {len(hr_files)} sharp images")
print(f"Prepared {len(files)} image pairs")
# Validate pairing correct?
if len(files) > 0:
    print(f"First image pair:\n  LR: {files[0][0]}\n  HR: {files[0][1]}")

# actual cropping
Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))

############ Prepare test data ####################
val_patch_size = 256
src = 'GOPRO_Large/test'
tar = 'Datasets/test/GoPro'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, '*', 'blur', '*.png')))
hr_files = natsorted(glob(os.path.join(src, '*', 'sharp', '*.png')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(test_files)(file_) for file_ in tqdm(files))