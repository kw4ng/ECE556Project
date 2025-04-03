import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Spaital Rearrangment Unit
class SpatialRearrangementUnit(nn.Module):
    def __init__(self, window_size, step_size):
        """
        window_size: local window size (assumed square), e.g. 4 means a 4x4 window.
        The step size is half of the window size.
        """
        super(SpatialRearrangementUnit, self).__init__()
        self.window_size = window_size
        self.step = step_size

    def rearrange_dimension(self, x, dim):
        """
        Rearrange features along specified dimension (width=3 or height=2).
        Input: x: tensor of shape (B, C, H, W)
        """
        chunk_size = self.window_size // 2
        chunks = list(x.split(chunk_size, dim=dim))

        # chunks = [01 23 45 67 89 1011]
        # padded chunks = [01 23 01 23 45 67 89 1011 89 1011]
        
        # Number of complete groups (excluding boundaries)
        num_chunks = (x.size(dim) - 2 * self.step) // chunk_size  # 12 / 2 = 6
        num_groups = num_chunks // 2  # 3??
        
        new_chunks = []
        # Process groups by combining alternate chunks
        for i in range(0, num_groups):

            # depending on the step size padding, the "first" group index of the original image will be different
            
            original_input_chunk_index = i * chunk_size + self.step // chunk_size

            left_index = (original_input_chunk_index - self.step // chunk_size)
            right_index = (original_input_chunk_index + 1) + self.step // chunk_size

            first = chunks[left_index]  # end of previous group
            second = chunks[right_index]  # start of next group
            new_chunks.append(torch.cat([first, second], dim=dim))

            # new chunks
            # [01 67 01 ]
            
        return torch.cat(new_chunks, dim=dim)

    def forward(self, x):
        """
        Apply width-direction rearrangement first, followed by height-direction.
        x: tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        chunk_size = self.window_size // 2

        # Create padding by using border chunks once
        left_pad = x[:, :, :, :self.step]  # left chunk
        right_pad = x[:, :, :, -self.step:]  # right chunk
        x_padded_w = torch.cat([left_pad, x, right_pad], dim=3)
        
        print("\nAfter width padding:")
        print(x_padded_w[0, 0])
        
        # Apply width rearrangement
        x_width = self.rearrange_dimension(x_padded_w, dim=3)
        
        # Now pad and rearrange in height direction
        top_pad = x_width[:, :, :self.step, :]  # top chunk
        bottom_pad = x_width[:, :, -self.step:, :]  # bottom chunk
        x_padded_h = torch.cat([top_pad, x_width, bottom_pad], dim=2)
        
        #print("\nAfter height padding:")
        #print(x_padded_h[0, 0])
        
        # Apply height rearrangement
        x_final = self.rearrange_dimension(x_padded_h, dim=2)
        
        return x_final

# Window Patition Unit
class WindowPartitioningUnit(nn.Module):
    """
    Splits the rearranged feature map into non-overlapping MxM windows.
    This unit assumes input shape: (B, C, H, W)
    """

    def __init__(self, window_size):
        super(WindowPartitioningUnit, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            windows: Tensor of shape (num_windows * B, C, M, M)
        """
        B, C, H, W = x.shape
        M = self.window_size

        assert H % M == 0 and W % M == 0, \
            f"Input height and width must be divisible by window size ({M})."

        # Reshape input into grid of windows
        x = x.view(B, C, H // M, M, W // M, M)
        # (B, C, num_h, M, num_w, M)

        # Rearrange dimensions to group each window
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # (B, num_h, num_w, C, M, M)

        # Flatten spatial window grid into batch
        windows = x.view(-1, C, M, M)
        # (B * num_windows, C, M, M)

        return windows

# Spatial Projection Unit with FC
class SpatialProjectionUnit(nn.Module):
    """
    Applies spatial projection to each local window using a fully connected (linear) layer
    WITHOUT gating. This summarizes the window into a learned representation.

    Input shape: (num_windows, C, H, W)
    Output shape: (num_windows, C, H, W) — same as input, but values transformed
    """

    def __init__(self, window_size):
        super(SpatialProjectionUnit, self).__init__()
        self.window_size = window_size # window size should be same with previous window size
        self.flatten_dim = window_size * window_size # calculate flattened dimension (eg. (16,1,4,4) will be (16,16))

        # Fully connected layer for projection
        self.fc = nn.Linear(self.flatten_dim, self.flatten_dim) # fc layer

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten each window
        x_flat = x.view(B, C, -1)  # shape: (B, C*H*W)
        #print("\nFlattened Input:")
        #print(x_flat)
        
        # Apply FC projection (No expand/shrink) {Maybe Wrong}
        x_proj = self.fc(x_flat)  # shape: (B, C*H*W)
        #print("\nProjected Output:")
        #print(x_proj)
        
        # Reshape back to original window shape (mainly for putting it in linear gating thus this form)
        x_out = x_proj.view(B, C, H, W)

        return x_out

# Window Merging Unit
class WindowMergingUnit(nn.Module):
    def __init__(self, window_size, original_height, original_width):
        super(WindowMergingUnit, self).__init__()
        self.window_size = window_size
        self.H = original_height
        self.W = original_width

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_windows * B, C, M, M)
        Returns:
            merged: Tensor of shape (B, C, H, W)
        """
        B_windows, C, M, M2 = x.shape
        assert M == self.window_size and M2 == self.window_size, "Window size mismatch"

        B = B_windows // ((self.H // M) * (self.W // M))
        assert B * (self.H // M) * (self.W // M) == B_windows, "Invalid input size for merging"

        # Reshape and permute back to (B, C, H, W)
        x = x.view(B, self.H // M, self.W // M, C, M, M)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, self.H, self.W)

        return x

import torch
import torch.nn as nn

class SpatialRearrangementRestorationUnit(nn.Module):
    def __init__(self, window_size, step_size):
        """
        window_size: local window size (assumed square), e.g. 4 means a 4×4 window.
        step_size: usually half of window_size (e.g. 2 for window_size=4).
        """
        super(SpatialRearrangementRestorationUnit, self).__init__()
        self.window_size = window_size
        self.step = step_size

    def inverse_rearrange_dimension(self, x, dim):
        """
        Inverse the rearrangement along the specified dimension.
        The forward rearrangement splits the padded tensor (of size original + 2*self.step)
        into chunks of size (window_size//2) and then, for each group, concatenates two chunks:
          left_index = i*chunk_size and right_index = (i*chunk_size + 1) + (self.step // chunk_size).
        Here we reconstruct a list of padded chunks of total length:
          total_chunks = (x.size(dim) + 2*self.step) // chunk_size.
        We fill the positions corresponding to the forward mapping and, for any missing slots,
        we fill as follows: if the missing index is 1 (i.e. the first pad) we copy index 0;
        if it is the second-to-last index, we copy the last index.
        """
        chunk_size = self.window_size // 2
        offset = self.step // chunk_size
        # Number of groups in the rearranged output along this dimension:
        num_groups = x.size(dim) // (2 * chunk_size)
        # The padded tensor was split into total_chunks chunks.
        total_chunks = (x.size(dim) + 2 * self.step) // chunk_size  # e.g. (16+8)//2 = 12

        padded_chunks = [None] * total_chunks

        # Split x into groups (each group has size 2*chunk_size)
        groups = list(x.split(2 * chunk_size, dim=dim))
        # groups: [1 2 7 8; 1 2 11 12; 5 6 15 16; 9 10 15 16]
        # i = 0 1 2 3
        # for i, group in enumerate(groups):
        #     first_half, second_half = group.split(chunk_size, dim=dim)
        #     # According to the forward mapping for step_size=2:
        #     #   original_input_chunk_index = i * chunk_size + (self.step // chunk_size)
        #     #   left_index = original_input_chunk_index - (self.step // chunk_size)
        #     #   right_index = (original_input_chunk_index + 1) + (self.step // chunk_size)
        #     left_index = i * chunk_size  # for step=2 and chunk_size=2, i*2: 0,2,4,6,...
        #     right_index = left_index + 3   # 0+3=3, 2+3=5, 4+3=7, 6+3=9
        #     padded_chunks[left_index] = first_half
        #     padded_chunks[right_index] = second_half

        for i, group in enumerate(groups):
            # Each group was split into two halves (each of size chunk_size)
            first_half, second_half = group.split(chunk_size, dim=dim)
            # Compute indices from parameters:
            left_index = i * chunk_size  # = original_input_chunk_index - offset
            right_index = i * chunk_size + 1 + 2 * offset  # = (original_input_chunk_index + 1) + offset
            padded_chunks[left_index] = first_half
            padded_chunks[right_index] = second_half

        # [1 2; X; 3 4; 5 6; 7 8; 9 10; 11 12; 13 14; X; 15 16]
        # padded chunks
        # [1 2; X; 1 2; X; 5 6; 7 8; 9 10; 11 12; X; 15 16; X; 15 16]

        # Fill in the missing chunks.
        for idx in range(total_chunks):
            if padded_chunks[idx] is None:
                print("IDX MISSING: ", idx)
                if idx < total_chunks / 2:
                    # use left boundary
                    padded_chunks[idx] = padded_chunks[idx - 1]
                else:
                    padded_chunks[idx] = padded_chunks[idx + 1]
                    
        # Reconstruct the padded tensor.
        restored_padded = torch.cat(padded_chunks, dim=dim)
        return restored_padded

    def forward(self, x):
        """
        Undo the spatial rearrangement along both dimensions.
        The inverse is applied in reverse order:
          1. Inverse rearrangement along the height (dim=2),
          2. Remove height padding,
          3. Inverse rearrangement along the width (dim=3),
          4. Remove width padding.
        """
        # Inverse height rearrangement.
        x_inv_h = self.inverse_rearrange_dimension(x, dim=2)
        # Remove height padding (assumes padding of self.step rows at top and bottom).
        x_unpad_h = x_inv_h[:, :, self.step: -self.step, :]
        # Inverse width rearrangement.
        x_inv_w = self.inverse_rearrange_dimension(x_unpad_h, dim=3)
        # Remove width padding (assumes padding of self.step columns at left and right).
        x_unpad_w = x_inv_w[:, :, :, self.step: -self.step]
        return x_unpad_w

# SRM block that intigrate all 5 previous class
class SRMBlock(nn.Module):
    def __init__(self, window_size, step_size, in_channels, original_height, original_width):
        """
        window_size: size of the local window (e.g., 4 for a 4x4 window).
        in_channels: number of channels of the input feature map.
        original_height, original_width: dimensions of the feature map after rearrangement.
        """
        super(SRMBlock, self).__init__()
        self.rearrangement = SpatialRearrangementUnit(window_size,step_size)
        self.partitioning = WindowPartitioningUnit(window_size)
        self.projection = SpatialProjectionUnit(window_size)
        self.merging = WindowMergingUnit(window_size, original_height, original_width)
        self.restoration = SpatialRearrangementRestorationUnit(window_size, step_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        # 1. Apply spatial rearrangement (with padding)
        x = self.rearrangement(x)
        # 2. Partition into non-overlapping windows
        x = self.partitioning(x)
        # 3. Apply spatial projection (window-based fully connected layer)
        x = self.projection(x)
        # 4. Merge windows back into a full feature map
        x = self.merging(x)
        # 5. Restore original spatial ordering (inverse rearrangement)
        x = self.restoration(x)
        return x
        
#######################Test Functions###################################
def create_test_input(size=16):
    # Create a 16x16 matrix with values 1-256
    values = np.arange(1, size*size + 1).reshape(size, size)
    # Convert to torch tensor and add batch and channel dimensions
    x = torch.FloatTensor(values).unsqueeze(0).unsqueeze(0)
    return x

def test_spatial_rearrangement():
    # Create input tensor (1, 1, 16, 16)
    x = create_test_input(16)
    print("Original input shape:", x.shape)
    print("\nOriginal input:")
    print(x[0, 0])

    # Test with window_size=4, step_size=2
    print("\n=== Testing with window_size=4, step_size=2 ===")
    sru_2 = SpatialRearrangementUnit(window_size=4, step_size=2)
    output_2 = sru_2(x)
    print("\nOutput shape:", output_2.shape)
    print("\nOutput with step_size=2:")
    print(output_2[0, 0])

    # Test restoration for step_size=2
    restoration_2 = SpatialRearrangementRestorationUnit(window_size=4, step_size=2)
    restored_2 = restoration_2(output_2)
    print("\nRestored output (step_size=2):")
    print(restored_2[0, 0])
    print("\nRestoration error (step_size=2):")
    print(torch.mean((x - restored_2).abs()))

    # Test with window_size=4, step_size=4
    print("\n=== Testing with window_size=4, step_size=4 ===")
    sru_4 = SpatialRearrangementUnit(window_size=4, step_size=4)
    output_4 = sru_4(x)
    print("\nOutput shape:", output_4.shape)
    print("\nOutput with step_size=4:")
    print(output_4[0, 0])

    # Test restoration for step_size=4
    restoration_4 = SpatialRearrangementRestorationUnit(window_size=4, step_size=4)
    restored_4 = restoration_4(output_4)
    print("\nRestored output (step_size=4):")
    print(restored_4[0, 0])
    print("\nRestoration error (step_size=4):")
    print(torch.mean((x - restored_4).abs()))

if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)
    test_spatial_rearrangement()


