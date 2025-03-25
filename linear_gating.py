import torch
import torch.nn as nn

print(torch.__version__)
print(torch.backends.mps.is_available())  

# Linear Gating is a mod

class LinearGating(nn.Module):
    # Initialize
    def __init__(self, dim):
        """
        dim: the number of channels/features per token
        """
        super().__init__()
        # This applies a linear transformation to the input features
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, C] where:
           B = batch size
           N = number of tokens (e.g., pixels or windows)
           C = number of channels (feature dimension)

        returns: Tensor of shape [B, N, C], gated version of input
        """
        # Eq given in paper G(x) = x * f_w,b(x)
        return x * self.fc(x)