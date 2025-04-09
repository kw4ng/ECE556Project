from mssr import MSSRNetwork
from channelMLP import ChannelMLP

import torch
import torch.nn as nn

class MSSRMLPBlock(nn.Module):
    def __init__(self, in_channels, window_size, final_height, final_width, step_sizes):
        """
        MSSR-MLP block using layer normalization and residual connections.
        
        Arguments:
          in_channels: number of channels (must be divisible by 3 for MSSRNetwork).
          window_size: the size of the window used in the MSSRNetwork.
          final_height: the height of feature maps in the MSSRNetwork.
          final_width: the width of feature maps in the MSSRNetwork.
          step_sizes: a list of three step sizes for the three SRM blocks inside MSSRNetwork.
        """
        super(MSSRMLPBlock, self).__init__()
        # LN is applied along the channel dimension; note that the input is expected [B, C, H, W].
        self.norm1 = nn.LayerNorm(in_channels)
        self.mssr = MSSRNetwork(window_size, in_channels, final_height, final_width, step_sizes)
        self.norm2 = nn.LayerNorm(in_channels)
        self.channel_mlp = ChannelMLP(in_channels)
    
    def forward(self, x):
        # print("Input shape:", x.shape)
        # x has shape [B, C, H, W]
        B, C, H, W = x.shape
        
        # Apply LayerNorm by permuting so that channels are the last dimension.
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # shape: [B, H, W, C]
        x_norm1 = self.norm1(x_perm)
        x_norm1 = x_norm1.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]
        
        # Pass through the MSSR module.
        mssr_out = self.mssr(x_norm1)
        # First residual connection.
        x1 = x + mssr_out
        
        # Second residual branch: apply layer normalization to x1.
        x1_perm = x1.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x1_norm = self.norm2(x1_perm)
        x1_norm = x1_norm.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Pass through the ChannelMLP.
        mlp_out = self.channel_mlp(x1_norm)
        x2 = x1 + mlp_out

        return x2