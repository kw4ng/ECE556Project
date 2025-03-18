import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Custom dataset class for loading a single image
class CustomImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path  # Single image path
        self.transform = transform

    def __len__(self):
        return 1  # We have only one image

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')  # Open image and convert to RGB

        if self.transform:
            image = self.transform(image)  # Apply transformation (resize, to tensor, etc.)
        
        return image

# Transformation for image preprocessing (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Reduce image size to 64x64
    transforms.ToTensor(),        # Convert image to tensor
])

# Path to the single image you want to use
image_path = r"C:\Users\anany\Downloads\ETH_BSRGAN.png"  # Make sure to use raw string for Windows paths

# Load the dataset (we are loading a single image)
dataset = CustomImageDataset(image_path=image_path, transform=transform)

# Create DataLoader for batching (batch size set to 1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Only one image, batch size = 1

# Define the Spatial Rearrangement MLP (SRM) model
class SpatialRearrangementMLP(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_size=64):  # Reduced hidden size
        super(SpatialRearrangementMLP, self).__init__()
        
        # Convolution layers for feature extraction (smaller hidden size)
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # Fully connected layers (MLP)
        self.mlp1 = nn.Linear(hidden_size * 64 * 64, 512)  # Adjusted for smaller image size
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, out_channels * 64 * 64)  # Output layer
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Feature extraction using convolutions
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for MLP
        x = x.view(x.size(0), -1)  # Flattening (batch_size, hidden_size * H * W)

        # MLP layers for spatial rearrangement
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.mlp3(x)

        # Reshape the output back to image format
        x = x.view(x.size(0), 3, 64, 64)  # (batch_size, channels, height, width)
        
        return x

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instance and move it to the selected device (GPU or CPU)
model = SpatialRearrangementMLP(in_channels=3, out_channels=3).to(device)

# Loss function (MSE) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
num_epochs = 20

# List to store loss values (for plotting loss curve later)
loss_values = []

# Training the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        # Move data to GPU if available
        batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()  # Zero gradients before backward pass
        outputs = model(batch)  # Pass through the model
        
        # Compute the loss (using MSE for image restoration)
        loss = criterion(outputs, batch)  # Compare output with input image (self-supervised)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss for monitoring
        running_loss += loss.item()

    # Store the loss for the current epoch
    loss_values.append(running_loss / len(dataloader))

    # Print the loss every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "srm_model.pth")

# Example for Inference (Restoring an Image)
# Load the trained model
model.load_state_dict(torch.load("srm_model.pth"))
model.eval()  # Set the model to evaluation mode

# Load the test image (same image used for training)
test_image = Image.open(r'C:\Users\anany\Downloads\ETH_BSRGAN.png').convert('RGB')

# Transform the image for prediction
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Match the size used in training
    transforms.ToTensor(),
])

test_image = test_transform(test_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Inference (no gradients needed during inference)
with torch.no_grad():  # No gradients are needed during inference
    restored_image = model(test_image)  # Get the restored image

# Convert tensor back to image
restored_image = restored_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Remove batch dimension and move to CPU
restored_image = (restored_image * 255).astype(np.uint8)  # Convert to uint8
restored_image = Image.fromarray(restored_image)  # Convert to PIL Image

# Show or save the restored image
restored_image.show()  # Display the image
