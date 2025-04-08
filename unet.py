import torch
import torch.nn as nn
import torch.nn.functional as F
from mssr import MSSRBlock  # assume this is your block with SRM inside

class Conv3x3LeakyReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.op(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.up(x)

class MSSRMLPUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=42, window_size=4, step_sizes=[2, 3, 4]):
        super().__init__()
        self.in_proj = Conv3x3LeakyReLU(in_channels, base_channels)

        # Encoder
        self.encoder1 = nn.Sequential(*[MSSRBlock(window_size, base_channels, 64, 64, step_sizes) for _ in range(2)])
        self.down1 = Downsample(base_channels)

        self.encoder2 = nn.Sequential(*[MSSRBlock(window_size, base_channels, 32, 32, step_sizes) for _ in range(4)])
        self.down2 = Downsample(base_channels)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[MSSRBlock(window_size, base_channels, 16, 16, step_sizes) for _ in range(4)])

        # Decoder
        self.up2 = Upsample(base_channels)
        self.decoder2 = nn.Sequential(*[MSSRBlock(window_size, base_channels, 32, 32, step_sizes) for _ in range(4)])

        self.up1 = Upsample(base_channels)
        self.decoder1 = nn.Sequential(*[MSSRBlock(window_size, base_channels, 64, 64, step_sizes) for _ in range(2)])

        # Final output conv to produce residual
        self.out_proj = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        input_img = x
        x = self.in_proj(x)

        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        # Decoder
        x = self.up2(x3) + x2
        x = self.decoder2(x)

        x = self.up1(x) + x1
        x = self.decoder1(x)

        # Output residual + skip
        residual = self.out_proj(x)
        return input_img + residual
