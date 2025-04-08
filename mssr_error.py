from srm import SRMBlock, SpatialRearrangementUnit, WindowPartitioningUnit, SpatialProjectionUnit, WindowMergingUnit, SpatialRearrangementRestorationUnit
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSSRNetwork(nn.Module):
    def __init__(self, window_size, in_channels, final_height, final_width, step_sizes):
        """
        Constructs an MSSR network that first projects channels, splits the feature map into 
        three parts, applies three separate SRM blocks (each with a different step size), 
        and then fuses the outputs.
        """
        super(MSSRNetwork, self).__init__()
        assert len(step_sizes) == 3, "Provide three step sizes."
        assert in_channels % 3 == 0, "in_channels must be divisible by 3 for splitting into equal parts."
        
        # Channel projection (here you can choose to maintain dimension or change it)
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # projecting channels
        self.gelu = nn.GELU()
        
        # Create three separate SRM branches for each channel chunk
        chunk_channels = in_channels//3
        self.srm1 = SRMBlock(window_size, chunk_channels, final_height, final_width, step_sizes[0])
        self.srm2 = SRMBlock(window_size, chunk_channels, final_height, final_width, step_sizes[1])
        self.srm3 = SRMBlock(window_size, chunk_channels, final_height, final_width, step_sizes[2])
        
        # Fuse the outputs from the three branches back together
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        
        # Split the feature map into 3 chunks along the channel dimension
        chunks = torch.chunk(x, 3, dim=1)
        print("CHUNK 0 shape:", chunks[0].shape)
        out1 = self.srm1(chunks[0])
        print("CHUNK 1 shape:", chunks[1].shape)
        out2 = self.srm2(chunks[1])
        print("CHUNK 2 shape:", chunks[2].shape)
        out3 = self.srm3(chunks[2])
        
        # Concatenate the outputs along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fuse(out)
        return out
