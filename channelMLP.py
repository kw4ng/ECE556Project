# FROM THE PAPER: 
# “We apply a channel-mixing MLP, which contains two fully connected layers with a GELU activation layer between them. 
# To make the MLP focus on local information, we add a depthwise convolution operation between the two FC layers.”

 # channels = 3 image is 256 x256

import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelMLP(nn.Module):
#     def __init__(self, channels):
#         super(ChannelMLP, self).__init__()
#         self.fc1 = nn.Linear(channels, channels * 2)
#         self.dwconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)  # Depthwise conv
#         self.gelu = nn.GELU()
#         self.fc2 = nn.Linear(channels * 2, channels)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1)  # B x H x W x C
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = x.permute(0, 3, 1, 2)  # B x C x H x W
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # B x H x W x C
#         x = self.fc2(x)
#         x = x.permute(0, 3, 1, 2)  # B x C x H x W
#         return x

class ChannelMLP(nn.Module):
    def __init__(self, channels):
        super(ChannelMLP, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.fc2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
