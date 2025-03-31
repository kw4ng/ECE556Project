import torch
import torch.nn as nn

class LinearGating(nn.Module):
    def __init__(self, dim, use_activation=False):
        """
        dim: number of input/output channels (C)
        use_activation: whether to apply sigmoid to f(x) before multiplying
        """
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.activation = nn.Sigmoid() if use_activation else nn.Identity()

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        returns: Gated tensor of shape [B, C, H, W]
        """
        assert x.ndim == 4, f"Expected input of shape [B, C, H, W], got {x.shape}"

        B, C, H, W = x.shape
        N = H * W # shrink dimension

        # Flatten spatial locations and permute to [B, N, C]
        x_reshaped = x.view(B, C, N).permute(0, 2, 1)

        # Apply linear gating
        gated = x_reshaped * self.activation(self.fc(x_reshaped))

        # Restore shape to [B, C, H, W]
        out = gated.permute(0, 2, 1).view(B, C, H, W)

        return out
