import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialRearrangementUnit(nn.Module):
    def __init__(self, window_size):
        """
        window_size: local window size (assumed square), e.g. 4 means a 4x4 window.
        The step size is half of the window size.
        """
        super(SpatialRearrangementUnit, self).__init__()
        self.window_size = window_size
        self.step = window_size // 2

    def rearrange_dimension(self, x, dim):
        """
        Rearrange features along specified dimension (width=3 or height=2).
        Input: x: tensor of shape (B, C, H, W)
        """
        chunk_size = self.window_size // 2
        chunks = list(x.split(chunk_size, dim=dim))
        
        # Number of complete groups (excluding boundaries)
        num_chunks = (x.size(dim) - 2 * chunk_size) // chunk_size
        num_groups = num_chunks // 2
        
        new_chunks = []
        # Process groups by combining alternate chunks
        for i in range(1, num_groups + 1):
            first = chunks[2 * i - 2]  # end of previous group
            second = chunks[2 * i + 1]  # start of next group
            new_chunks.append(torch.cat([first, second], dim=dim))
            
        return torch.cat(new_chunks, dim=dim)

    def forward(self, x):
        """
        Apply width-direction rearrangement first, followed by height-direction.
        x: tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        chunk_size = self.window_size // 2

        # Create padding by using border chunks once
        left_pad = x[:, :, :, :chunk_size]  # left chunk
        right_pad = x[:, :, :, -chunk_size:]  # right chunk
        x_padded_w = torch.cat([left_pad, x, right_pad], dim=3)
        
        print("\nAfter width padding:")
        print(x_padded_w[0, 0])
        
        # Apply width rearrangement
        x_width = self.rearrange_dimension(x_padded_w, dim=3)
        
        # Now pad and rearrange in height direction
        top_pad = x_width[:, :, :chunk_size, :]  # top chunk
        bottom_pad = x_width[:, :, -chunk_size:, :]  # bottom chunk
        x_padded_h = torch.cat([top_pad, x_width, bottom_pad], dim=2)
        
        print("\nAfter height padding:")
        print(x_padded_h[0, 0])
        
        # Apply height rearrangement
        x_final = self.rearrange_dimension(x_padded_h, dim=2)
        
        return x_final

if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)

    # Test case 1: 4x4 input with window_size=2
    dummy_input = torch.arange(1, 17).reshape(1, 1, 4, 4).float()
    print("Test Case 1 - 4x4 input, window_size=2")
    print("Input:")
    print(dummy_input)

    srm = SpatialRearrangementUnit(window_size=2)
    output = srm(dummy_input)
    print("\nOutput:")
    print(output)

    # Test case 2: 16x16 input with window_size=4
    dummy_input_large = torch.arange(1, 257).reshape(1, 1, 16, 16).float()
    print("\nTest Case 2 - 16x16 input, window_size=4")
    print("Input shape:", dummy_input_large.shape)
    
    srm_large = SpatialRearrangementUnit(window_size=4)
    output_large = srm_large(dummy_input_large)
    print("Output shape:", output_large.shape)
    
    # Print middle 8x8 squares
    print("\nMiddle 8x8 of input:")
    print(dummy_input_large[0, 0, 4:12, 4:12])
    print("\nMiddle 8x8 of output:")
    print(output_large[0, 0, 4:12, 4:12])

    print("all input")
    print(dummy_input_large)
    print("all output")
    print(output_large)
