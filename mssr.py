from srm import SRMBlock, SpatialRearrangementUnit, WindowPartitioningUnit, SpatialProjectionUnit, WindowMergingUnit, SpatialRearrangementRestorationUnit
import torch
import torch.nn as nn
import torch.nn.functional as F

# MSSR Block (One SRM Block) ------------
class MSSRBlock(nn.Module):
    def __init__(self, window_size, in_channels, final_height, final_width, step_size):
        """
        One SRM block performing rearrangement, window partitioning, projection,
        merging, and restoration. The final output is resized/cropped to final_height x final_width.
        """
        super(MSSRBlock, self).__init__()
        self.final_height = final_height
        self.final_width = final_width
        self.window_size = window_size
        self.step_size = step_size
        self.rearrangement = SpatialRearrangementUnit(window_size, step_size)
        self.partitioning = WindowPartitioningUnit(window_size)
        self.projection = SpatialProjectionUnit(in_channels, window_size)
        self.merging = WindowMergingUnit(window_size)
        self.restoration = SpatialRearrangementRestorationUnit(window_size, step_size)

    def forward(self, x):
        # Step 1: Spatial rearrangement.
        x = self.rearrangement(x)
        B, C, H_new, W_new = x.shape

        # Pad if needed so H_new and W_new are divisible by window_size.
        pad_h = (self.window_size - H_new % self.window_size) % self.window_size
        pad_w = (self.window_size - W_new % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_new += pad_h
            W_new += pad_w

        # Step 2: Partition into windows.
        x_windows = self.partitioning(x)
        # Step 3: Projection (MLP) on each window.
        x_proj = self.projection(x_windows)
        # Step 4: Merge windows back.
        x_merged = self.merging(x_proj, H_new, W_new)
        # Step 5: Restoration.
        x_restored = self.restoration(x_merged)
        # Crop back if padding was added.
        x_cropped = x_restored[:, :, :self.final_height, :self.final_width]
        # Resize to ensure final dimensions.
        x_final = F.interpolate(x_cropped, size=(self.final_height, self.final_width), mode='bilinear', align_corners=False)
        return x_final

# ----------- MSSR Network with 3 SRM Blocks ------------
# class MSSRNetwork(nn.Module):
#     def __init__(self, window_size, in_channels, final_height, final_width, step_sizes):
#         """
#         Constructs a network with three SRM blocks.
#         Each block uses the same window size but a different step size.
#         """
#         super(MSSRNetwork, self).__init__()
#         assert len(step_sizes) == 3, "Provide three step sizes."
#         self.blocks = nn.ModuleList([
#             MSSRBlock(window_size, in_channels, final_height, final_width, step_sizes[0]),
#             MSSRBlock(window_size, in_channels, final_height, final_width, step_sizes[1]),
#             MSSRBlock(window_size, in_channels, final_height, final_width, step_sizes[2])
#         ])

#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x

class MSSRNetwork(nn.Module):
    def __init__(self, window_size, in_channels, final_height, final_width, step_sizes):
        """
        Constructs a network with three SRM blocks.
        Each block uses the same window size but a different step size.
        """
        super(MSSRNetwork, self).__init__()
        assert len(step_sizes) == 3, "Provide three step sizes."

        self.in_channels = in_channels
        self.blocks = nn.ModuleList([
            SRMBlock(window_size, in_channels // 3, final_height, final_width, s)
            for s in step_sizes
        ])

        self.pre_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU()
        )
        self.post_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Channel projection + GELU
        x = self.pre_proj(x)

        # Split along channel dimension into 3 parts
        splits = torch.chunk(x, 3, dim=1)
        assert len(splits) == 3, "Input must be divisible by 3 channels"

        # Apply each SRM block to a split
        out_parts = [block(part) for block, part in zip(self.blocks, splits)]

        # Concatenate along channel dim
        x_cat = torch.cat(out_parts, dim=1)

        # Final channel projection to fuse multiscale features
        x_fused = self.post_proj(x_cat)
        return x_fused

