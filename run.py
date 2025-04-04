import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt  # Add this for displaying the image

from srm import SRMBlock, SpatialRearrangementUnit, WindowPartitioningUnit, SpatialProjectionUnit, WindowMergingUnit, SpatialRearrangementRestorationUnit
from mssr import MSSRBlock, MSSRNetwork 
from channelMLP import ChannelMLP  
from linear_gating import LinearGating 

def main(): 

    # Create a test input
    print("Starting processing for test_image_1...")
    
    # Path to your test image
    image_path = 'images/test_image_1.png'  # Adjust path as needed
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert BGR (OpenCV) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to match model expectations (optional but important)
    img = cv2.resize(img, (256, 256))

    # Convert [H, W, C] → [C, H, W] → [1, C, H, W]
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    # Normalize to [0, 1]
    # img_tensor = img_tensor / 255.0

    print(f"Input tensor shape: {img.shape}")
    

    print("\n=== Running MSSR (which runs SRM which runs linear gating)===")
    mssr_network = MSSRNetwork(
        window_size=4,  # Example window size
        in_channels=3,  # Number of input channels (e.g., RGB)
        final_height=256,  # Desired output height
        final_width=256,  # Desired output width
        step_sizes=[2, 4, 8]  # Example step sizes for the three blocks
    )  # Adjust parameters as needed
    mssr_output = mssr_network(img_tensor)
    
    print(f"After MSSR block: {mssr_output.shape}")
    
    if mssr_output.is_cuda:
        mssr_output_cpu = mssr_output.cpu()
    else:
        mssr_output_cpu = mssr_output
    
    mssr_img = mssr_output_cpu.detach().squeeze(0).permute(1, 2, 0).numpy()
    mssr_img = np.clip(mssr_img * 255, 0, 255).astype(np.uint8)
    
    # Save the MSSR intermediate output
    mssr_path = 'mssr_output.png'
    cv2.imwrite(mssr_path, cv2.cvtColor(mssr_img, cv2.COLOR_RGB2BGR))
    print(f"MSSR output image saved to {mssr_path}")

    # Step 3: Apply ChannelMLP
    print("\n=== Running ChannelMLP ===")
    channel_mlp = ChannelMLP(channels=3)  # Adjust parameters as needed
    final_output = channel_mlp(mssr_output)
    print(f"After ChannelMLP: {final_output.shape}")

    if final_output.is_cuda:
        final_output = final_output.cpu()
        
    # Detach from computation graph and convert to numpy
    output_img = final_output.detach().squeeze(0).permute(1, 2, 0).numpy()
    
    # Scale back to 0-255 range
    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
    
    # Save the output image
    output_path = 'processed_test_image_1.png'
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Processed image saved to {output_path}")
    
    # Display the original and processed images side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')
        
    plt.subplot(1, 3, 2)
    plt.title('After MSSR')
    plt.imshow(mssr_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Processed Image')
    plt.imshow(output_img)
    plt.axis('off')
    
    plt.savefig('comparison.png')
    plt.show()
    
    
    print("\nPipeline execution completed!")





if __name__ == "__main__":
    main()