import torch
import torch.nn as nn
from mssr_mlp import MSSRMLPBlock

# -------------------------------
# Helper functions: building stages and up/down-sampling modules.
# -------------------------------
def make_mssr_stage(in_channels, num_blocks, window_size, final_height, final_width, step_sizes):
    layers = []
    for block in range(num_blocks):
        layers.append(MSSRMLPBlock(in_channels, window_size, final_height, final_width, step_sizes))
    return nn.Sequential(*layers)

class Downsample(nn.Module):
    """3×3 convolution with stride 2 for downsampling."""
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """3×3 convolution followed by PixelShuffle for upsampling."""
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv(x)
        return self.pixel_shuffle(x)

# -------------------------------
# U-Net with integrated MSSR-MLP blocks (MSSR-MLP-B configuration)
# -------------------------------
class UNetMSSRMLP_B(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=42,
                 enc_depths=[2, 4, 12], bottleneck_depth=4, dec_depths=[12, 4, 2],
                 mssr_window_size=4, mssr_step_sizes=[0, 2, 4], input_size=256):
        """
        U-Net architecture integrated with MSSR-MLP blocks following the MSSR-MLP-B config.
        
        Args:
          in_channels: number of input channels.
          out_channels: number of output channels.
          base_channels: initial channel count (42 for MSSR-MLP-B).
          enc_depths: number of MSSRMLPBlock blocks in each encoder stage.
          bottleneck_depth: number of blocks in the bottleneck stage.
          dec_depths: number of MSSRMLPBlock blocks in each decoder stage (mirrored).
          mssr_window_size: window size parameter for MSSRMLPBlock.
          mssr_step_sizes: list of three step sizes for MSSRMLPBlock.
          input_size: spatial resolution (assumes square input) of the feature maps at stage 1.
        """
        super(UNetMSSRMLP_B, self).__init__()
        
        # Initial convolution: from RGB to base_channels.
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder Stage 1
        # Spatial dims: input_size x input_size; channels: base_channels (42).
        self.encoder1 = make_mssr_stage(base_channels, enc_depths[0],
                                        mssr_window_size, input_size, input_size, mssr_step_sizes)
        self.down1 = Downsample(base_channels, base_channels * 2)  # 42 -> 84; dims become input_size/2
        
        # Encoder Stage 2
        self.encoder2 = make_mssr_stage(base_channels * 2, enc_depths[1],
                                        mssr_window_size, input_size // 2, input_size // 2, mssr_step_sizes)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)  # 84 -> 168; dims: input_size/4
        
        # Encoder Stage 3
        self.encoder3 = make_mssr_stage(base_channels * 4, enc_depths[2],
                                        mssr_window_size, input_size // 4, input_size // 4, mssr_step_sizes)
        self.down3 = Downsample(base_channels * 4, base_channels * 8)  # 168 -> 336; dims: input_size/8
        
        # Bottleneck Stage
        self.bottleneck = make_mssr_stage(base_channels * 8, bottleneck_depth,
                                          mssr_window_size, input_size // 8, input_size // 8, mssr_step_sizes)
        
        # Decoder Stage 3
        self.up3 = Upsample(base_channels * 8, base_channels * 4)  # 336 -> 168; dims: input_size/4
        self.decoder3 = make_mssr_stage(base_channels * 4, dec_depths[0],
                                        mssr_window_size, input_size // 4, input_size // 4, mssr_step_sizes)
        
        # Decoder Stage 2
        self.up2 = Upsample(base_channels * 4, base_channels * 2)  # 168 -> 84; dims: input_size/2
        self.decoder2 = make_mssr_stage(base_channels * 2, dec_depths[1],
                                        mssr_window_size, input_size // 2, input_size // 2, mssr_step_sizes)
        
        # Decoder Stage 1
        self.up1 = Upsample(base_channels * 2, base_channels)  # 84 -> 42; dims: input_size
        self.decoder1 = make_mssr_stage(base_channels, dec_depths[2],
                                        mssr_window_size, input_size, input_size, mssr_step_sizes)
        
        # Final output convolution: produces a residual image to be added to the input.
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Initial feature extraction.
        # print("1")
        x0 = self.input_conv(x)
        # print("2")
        # print("x0 shape:", x0.shape)
        # Encoder with skip connections.
        e1 = self.encoder1(x0)             # Stage 1 output: [B, 42, input_size, input_size]
        # print("3")
        d1 = self.down1(e1)                # [B, 84, input_size/2, input_size/2]
        # print("4")
        e2 = self.encoder2(d1)             # [B, 84, input_size/2, input_size/2]
        # print("5")
        d2 = self.down2(e2)                # [B, 168, input_size/4, input_size/4]
        # print("6")
        e3 = self.encoder3(d2)             # [B, 168, input_size/4, input_size/4]
        # print("7")
        d3 = self.down3(e3)                # [B, 336, input_size/8, input_size/8]
        # print("8")
        # print("d3 shape:", d3.shape)
        # Bottleneck.
        b = self.bottleneck(d3)            # [B, 336, input_size/8, input_size/8]
        # print("9")
        # Decoder with skip connections.
        u3 = self.up3(b)                   # Upsample to [B, 168, input_size/4, input_size/4]
        d3_out = self.decoder3(u3 + e3)     # Merge with encoder stage 3
        
        u2 = self.up2(d3_out)              # Upsample to [B, 84, input_size/2, input_size/2]
        d2_out = self.decoder2(u2 + e2)     # Merge with encoder stage 2
        
        u1 = self.up1(d2_out)              # Upsample to [B, 42, input_size, input_size]
        d1_out = self.decoder1(u1 + e1)     # Merge with encoder stage 1
        
        # Output: produce residual image and add it to input.
        residual = self.output_conv(d1_out)
        out = x + residual
        return out
    