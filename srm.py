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

    def __init__(self, in_channels, window_size):
        super(SpatialProjectionUnit, self).__init__()
        self.in_channels = in_channels # channel in should always be 3 channel color 
        self.window_size = window_size # window size should be same with previous window size
        self.flatten_dim = in_channels * window_size * window_size # calculate flattened dimension (eg. (16,3,4,4) will be (16,48))

        # Fully connected layer for projection
        self.fc = nn.Linear(self.flatten_dim, self.flatten_dim) # fc layer

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels and H == W == self.window_size, \
            f"Expected input shape (B, {self.in_channels}, {self.window_size}, {self.window_size}), got {x.shape}"

        # Flatten each window
        x_flat = x.view(B, -1)  # shape: (B, C*H*W)
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

# Rearrangment Unit (pretty much a altered version of original rearrangment code)
class SpatialRearrangementRestorationUnit(nn.Module):
    def __init__(self, window_size):
        """
        window_size: local window size (assumed square), e.g. 4 means a 4x4 window.
        The step size is half of the window size.
        """
        super(SpatialRearrangementRestorationUnit, self).__init__()
        self.window_size = window_size
        self.step = window_size // 2
   
    def restore_dimension(self, x, dim):
        """
        Restores the original ordering along a given dimension by processing
        all groups and reintroducing the boundary chunks.
        """
        chunk_size = self.window_size // 2
        # Split the tensor along the dimension.
        chunks = list(x.split(chunk_size, dim=dim))
        # Assume that the original forward unit padded by taking the first and last chunks
        # as boundaries. So we save them:
        left_boundary = chunks[0]
        right_boundary = chunks[-1]
        # The remaining chunks (the interior ones) are assumed to have been processed in groups.
        interior_chunks = chunks[1:-1]
        # Compute number of groups from the interior:
        num_chunks = len(interior_chunks)  # this should be even
        num_groups = num_chunks // 2

        new_chunks = []
        # Process each interior group.
        # In the forward unit each group was formed by concatenating:
        #   [chunks[2*i - 2], chunks[2*i + 1]]
        # So for inversion we swap the order.
        for i in range(1, num_groups + 1):
            # Note: interior_chunks[0] corresponds to chunks[1] from the original,
            # interior_chunks[-1] corresponds to chunks[-2]
            first = interior_chunks[2*i - 2]  # originally from the left part of the group
            second = interior_chunks[2*i - 1]  # originally from the right part of the group
            # To invert the forward permutation, swap their order.
            new_chunks.append(torch.cat([second, first], dim=dim))
        # Now, reintroduce the boundary chunks.
        # You can choose to prepend left_boundary and append right_boundary,
        # or merge them in a way that restores the full dimension.
        # For example, if the forward pass simply removed these boundaries,
        # we can simply:
        restored = torch.cat([left_boundary] + new_chunks + [right_boundary], dim=dim)
        return restored


    def forward(self, x):
        """
        Apply width-direction rearrangement first, followed by height-direction.
        x: tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        chunk_size = self.window_size // 2
        
        # Apply width rearrangement
        x_width = self.restore_dimension(x, dim=3)
        
        #print("\nAfter height padding:")
        #print(x_padded_h[0, 0])
        
        x_restore = self.restore_dimension(x_width, dim=2)
        
        return x_restore

# SRM block that intigrate all 5 previous class
class SRMBlock(nn.Module):
    def __init__(self, window_size, in_channels, original_height, original_width):
        """
        window_size: size of the local window (e.g., 4 for a 4x4 window).
        in_channels: number of channels of the input feature map.
        original_height, original_width: dimensions of the feature map after rearrangement.
        """
        super(SRMBlock, self).__init__()
        self.rearrangement = SpatialRearrangementUnit(window_size)
        self.partitioning = WindowPartitioningUnit(window_size)
        self.projection = SpatialProjectionUnit(in_channels, window_size)
        self.merging = WindowMergingUnit(window_size, original_height, original_width)
        self.restoration = SpatialRearrangementRestorationUnit(window_size)
        
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

    print("Original input shape:", x.shape)
    print("\nOriginal input:")
    print(x[0, 0])

    # Test with window_size=4, step_size=4
    print("\n=== Testing with window_size=4, step_size=4 ===")
    sru_4 = SpatialRearrangementUnit(window_size=4, step_size=4)
    output_4 = sru_4(x)
    print("\nOutput shape:", output_4.shape)
    print("\nOutput with step_size=4:")
    print(output_4[0, 0])

if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)
    test_spatial_rearrangement()


