import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision import transforms


from unet import UNetMSSRMLP_B

def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred, norm='ortho')
    target_fft = torch.fft.fft2(target, norm='ortho')
    return torch.mean(torch.abs(pred_fft - target_fft))

def train_mssr_mlp(model, dataloader, num_epochs=3000, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    l1_loss = nn.L1Loss()
    lambda_freq = 0.1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress:
            input_img, target_img = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            output = model(input_img)

            loss_content = l1_loss(output, target_img)
            loss_freq = frequency_loss(output, target_img)
            loss = loss_content + lambda_freq * loss_freq

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(dataloader):.6f}")



# -------------------------------
# Custom Dataset for the GoPro dataset.
# -------------------------------
class GoProDataset(Dataset):
    def __init__(self, degraded_dir, ground_truth_dir, transform=None):
        """
        Args:
          degraded_dir: Directory for degraded (input) images.
          ground_truth_dir: Directory for ground truth images.
          transform: PyTorch transforms (e.g. random crop, flip, etc.)
        """
        self.degraded_dir = degraded_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.degraded_files = sorted(os.listdir(degraded_dir))
        self.gt_files = sorted(os.listdir(ground_truth_dir))
        assert len(self.degraded_files) == len(self.gt_files), "Mismatch in number of images."

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, idx):
        degraded_path = os.path.join(self.degraded_dir, self.degraded_files[idx])
        gt_path = os.path.join(self.ground_truth_dir, self.gt_files[idx])
        degraded_img = Image.open(degraded_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.transform:
            degraded_img = self.transform(degraded_img)
            gt_img = self.transform(gt_img)

        return degraded_img, gt_img

# -------------------------------
# Frequency loss function
# -------------------------------
def frequency_loss(output, target):
    # Compute FFT2 for each image
    fft_output = torch.fft.fft2(output)
    fft_target = torch.fft.fft2(target)
    # Use the magnitude (absolute value) of the FFT results
    mag_output = torch.abs(fft_output)
    mag_target = torch.abs(fft_target)
    # Compute the L1 difference between the magnitudes
    return torch.mean(torch.abs(mag_output - mag_target))

# -------------------------------
# Define transforms for training (random crop to 256x256, augmentation, etc.)
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to [0, 1] FloatTensor
])

# Replace these paths with your actual training directories.
degraded_dir = 'Datasets/train/GoPro/input_crops'
ground_truth_dir = 'Datasets/train/GoPro/target_crops'

train_dataset = GoProDataset(degraded_dir, ground_truth_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# -------------------------------
# Instantiate the model (UNet with MSSR-MLP-B configuration)
# -------------------------------
# Import or define the integrated model UNetMSSRMLP_B from the previous code.
# (Ensure that UNetMSSRMLP_B, MSSRMLPBlock, MSSRNetwork, SRMBlock, ChannelMLP,
#  Downsample, and Upsample are defined as in the integrated code provided earlier.)
#
# For example:
# from your_model_file import UNetMSSRMLP_B

# Here we assume that UNetMSSRMLP_B is already defined.
# Use the following configuration for MSSR-MLP-B:
#   Base channels: 42
#   Encoder depths: [2, 4, 12]
#   Bottleneck depth: 4
#   Decoder depths: [12, 4, 2]
#   MSSR window size: 4
#   MSSR step sizes: [1, 2, 3]
#   Input image size: 256 (assumed square)
model = UNetMSSRMLP_B(
    in_channels=3,
    out_channels=3,
    base_channels=42,
    enc_depths=[2, 4, 12],
    bottleneck_depth=4,
    dec_depths=[12, 4, 2],
    mssr_window_size=4,
    mssr_step_sizes=[1, 2, 3],
    input_size=256
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -------------------------------
# Define loss functions and optimizer
# -------------------------------
l1_loss_fn = nn.L1Loss()  # Content Loss
lambda_freq = 0.1         # Frequency loss weight

def total_loss_fn(output, target):
    content_loss = l1_loss_fn(output, target)
    freq_loss_val = frequency_loss(output, target)
    total = content_loss + lambda_freq * freq_loss_val
    return total

def main(): 
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    # Use cosine annealing to drop LR from 1e-3 to 1e-6 over 3000 epochs.
    scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-6)

    # -------------------------------
    # Training loop
    # -------------------------------
    num_epochs = 3000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for degraded, target in train_loader:
            degraded, target = degraded.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(degraded)
            loss = total_loss_fn(output, target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * degraded.size(0)
        
        avg_loss = running_loss / len(train_dataset)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")
            
    # Optionally, save the model after training
    torch.save(model.state_dict(), "gopro_mssrmlp_b.pth")


if __name__ == "__main__":
    main()