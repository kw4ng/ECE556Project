import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch import amp
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt  # for loss graph

from PIL import Image

# Import the model from your "unet.py" file.
from unet import UNetMSSRMLP_B


# -------------------------------
# Dataset for GoPro cropped images.
# -------------------------------
class GoProCropDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        assert len(self.input_files) == len(self.target_files), "Mismatch between input and target images."

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img


# -------------------------------
# Loss Functions
# -------------------------------
def frequency_loss(output, target):
    fft_output = torch.fft.fft2(output)
    fft_target = torch.fft.fft2(target)
    mag_output = torch.abs(fft_output)
    mag_target = torch.abs(fft_target)
    return torch.mean(torch.abs(mag_output - mag_target))


def total_loss_fn(output, target, lambda_freq=0.1):
    l1_loss = nn.L1Loss()(output, target)
    freq_loss = frequency_loss(output, target)
    return l1_loss + lambda_freq * freq_loss


def train_one_epoch(model, dataloader, optimizer, device, lambda_freq):
    model.train()
    running_loss = 0.0
    # Create a progress bar wrapping the dataloader.
    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for i, (degraded, target) in progress:
        degraded, target = degraded.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(degraded)
        loss = total_loss_fn(output, target, lambda_freq)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * degraded.size(0)
        # Compute current average loss for the batches processed so far.
        batch_loss = running_loss / ((i + 1) * degraded.size(0))

        # Update the progress bar's postfix to display the current loss.
        progress.set_postfix(loss=batch_loss)

        # Optionally, print the loss every 10 batches.
        # if (i + 1) % 10 == 0:
            # print(f"Batch {i + 1}/{len(dataloader)}: Current loss: {batch_loss:.6f}")

    return running_loss / len(dataloader.dataset)



# -------------------------------
# Main Training Script
# -------------------------------
def main():
    # Prompt for hyperparameters
    print("training 4/20")
    num_epochs = int(input("Enter number of epochs: "))
    batch_size = int(input("Enter batch size: "))

    print(f"Training for {num_epochs} epochs with batch size {batch_size}")

    learning_rate = 1e-3
    input_size = 256
    lambda_freq = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Device name:", torch.cuda.get_device_name(device) if device.type == 'cuda' else "CPU")

    # Directories for training crops
    train_input_dir = os.path.join("Datasets", "train", "GoPro", "input_crops")
    train_target_dir = os.path.join("Datasets", "train", "GoPro", "target_crops")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = GoProCropDataset(train_input_dir, train_target_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = UNetMSSRMLP_B(
        in_channels=3,
        out_channels=3,
        base_channels=42,
        enc_depths=[2, 4, 12],
        bottleneck_depth=4,
        dec_depths=[12, 4, 2],
        mssr_window_size=4,
        mssr_step_sizes=[0, 2, 4],
        input_size=input_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    epoch_losses = []  # track loss for plotting
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, lambda_freq)
        scheduler.step()
        epoch_losses.append(train_loss)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.6f}")

    # Determine final loss and build filenames
    final_loss = epoch_losses[-1]
    model_filename = f"gopro_mssrmlp_b_e{num_epochs}_bs{batch_size}_loss{final_loss:.6f}.pth"
    plot_filename = f"loss_plot_e{num_epochs}_bs{batch_size}_loss{final_loss:.6f}.png"

    # Save the final model state
    torch.save(model.state_dict(), model_filename)
    print(f"Saved model to {model_filename}")

    # Plot and save the loss graph
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig(plot_filename)
    print(f"Saved loss plot to {plot_filename}")


if __name__ == "__main__":
    main()
