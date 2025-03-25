import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialRearrangementUnit(nn.Module):
    def __init__(self, window_size):
        """
        window_size: local window size (assumed square), e.g. 4 means a 4x4 window.
        The step size is assumed to be half of the window size.
        """
        super(SpatialRearrangementUnit, self).__init__()
        self.window_size = window_size
        self.step = window_size // 2

    def width_rearrangement(self, x):
        """
        Rearrange features along the width dimension.
        Input:
            x: tensor of shape (B, C, H, W)
        Process:
            - Split the width dimension into chunks of size (window_size//2)
            - Group every two adjacent chunks (to form a block of width equal to window_size)
            - For each group (except at boundaries), replace the current group by
              using chunks from the previous and next positions.
        """
        B, C, H, W = x.shape
        chunk_size = self.window_size // 2
        # For simplicity, assume W is divisible by chunk_size; otherwise, pad appropriately.
        num_chunks = W // chunk_size
        # Split along width: list of tensors each of shape (B, C, H, chunk_size)
        x_chunks = list(x.split(chunk_size, dim=3))
        print("x_chunks")
        print(x_chunks)
        new_chunks = []
        # Process first group (boundary: keep original)
        if num_chunks >= 2:
            first_group = torch.cat([x_chunks[0], x_chunks[1]], dim=3)
            new_chunks.append(first_group)
        # Process inner groups: for group i (starting from 1 up to num_groups-2)
        # each group is replaced by chunks from one group earlier and one group later.
        num_groups = num_chunks // 2  # each group has two chunks
        for i in range(1, num_groups - 1):
            # Use chunk from previous group's first half and next group's second half
            left = x_chunks[2 * i - 1]  # end of previous group
            right = x_chunks[2 * i + 1]  # start of next group
            group = torch.cat([left, right], dim=3)
            new_chunks.append(group)
        # Process last group (boundary: keep original)
        if num_chunks % 2 == 0 and num_chunks >= 2:
            last_group = torch.cat([x_chunks[-2], x_chunks[-1]], dim=3)
            new_chunks.append(last_group)
        # Concatenate all groups along width.
        x_rearranged = torch.cat(new_chunks, dim=3)
        print("x_rearranged")
        print(x_rearranged)
        return x_rearranged

    def height_rearrangement(self, x):
        """
        Rearrange features along the height dimension in a similar manner as width.
        Input:
            x: tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        chunk_size = self.window_size // 2
        num_chunks = H // chunk_size
        x_chunks = list(x.split(chunk_size, dim=2))
        new_chunks = []
        # Process first group (boundary)
        if num_chunks >= 2:
            first_group = torch.cat([x_chunks[0], x_chunks[1]], dim=2)
            new_chunks.append(first_group)
        # Process inner groups
        num_groups = num_chunks // 2
        for i in range(1, num_groups - 1):
            top = x_chunks[2 * i - 1]
            bottom = x_chunks[2 * i + 1]
            group = torch.cat([top, bottom], dim=2)
            new_chunks.append(group)
        # Process last group (boundary)
        if num_chunks % 2 == 0 and num_chunks >= 2:
            last_group = torch.cat([x_chunks[-2], x_chunks[-1]], dim=2)
            new_chunks.append(last_group)
        # Concatenate all groups along height.
        x_rearranged = torch.cat(new_chunks, dim=2)
        return x_rearranged

    def forward(self, x):
        """
        Apply width-direction rearrangement first, followed by height-direction rearrangement.
        x: tensor of shape (B, C, H, W)
        """
        # (Optional) Pad boundaries if needed so that H and W are divisible by window_size//2.
        # Here we assume that the input size already meets this condition.
        x_width = self.width_rearrangement(x)
        x_rearranged = self.height_rearrangement(x_width)
        return x_rearranged

# Example usage:
if __name__ == '__main__':
    # Create a dummy input tensor with batch size 1, 3 channels, height 8, width 8.
    # For a window_size of 4, we require H and W to be multiples of (window_size//2)=2.
    # dummy_input = torch.randn(1, 3, 8, 8)
    # create input that is 4x4 and labeled from 1 to 16
    dummy_input = torch.arange(1, 17).reshape(1, 1, 4, 4).float()
    print("dummy_input")
    print(dummy_input)

    srm = SpatialRearrangementUnit(window_size=2)
    output = srm(dummy_input)
    print("output")
    print(output)


    # srm = SpatialRearrangementUnit(window_size=4)
    # output = srm(dummy_input)
    # print("Input shape:", dummy_input.shape)
    # print("Output shape:", output.shape)
