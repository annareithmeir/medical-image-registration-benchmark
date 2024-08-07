from typing import List, Union

import torch
import torch.nn.functional as F


def compute_grid(image_size: torch.Size) -> torch.Tensor:
    """
    Compute a normalized grid for a given image size.

    @param image_size: The size of the image. Should be a list of three integers.

    @return: A tensor representing the normalized grid.
    """

    nz = image_size[0]
    ny = image_size[1]
    nx = image_size[2]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)
    z = torch.linspace(-1, 1, steps=nz)

    x = x.expand(ny, -1).expand(nz, -1, -1)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(4)
    y.unsqueeze_(0).unsqueeze_(4)
    z.unsqueeze_(0).unsqueeze_(4)

    return torch.cat((x, y, z), 4)


def deform_image(image: torch.Tensor,
                 displacement: torch.Tensor) -> torch.Tensor:

    grid = compute_grid(image.shape[2:])  # torch.Size([1, 2, 2, 2, 3])

    grid = torch.zeros_like(grid)

    return F.grid_sample(image, displacement + grid, mode='bilinear')


# image = torch.rand(1, 1, 2, 2, 2)
# zero_def = torch.zeros(1, 2, 2, 2, 3)

# deformed = deform_image(image, zero_def)

# print((deformed-image).sum())


# Assuming you have these tensors:
# Shape: (batch, channel, depth, height, width)
volume = torch.randn(1, 1, 10, 20, 30)
# Shape: (batch, 3, depth, height, width)
displacement_field = torch.randn(1, 3, 10, 20, 30)
displacement_field = torch.zeros_like(displacement_field)

# Create a grid of normalized coordinates
grid = F.affine_grid(torch.eye(3, 4).unsqueeze(
    0), volume.shape, align_corners=False)

# Add the displacement field to the grid
deformed_grid = grid + displacement_field.permute(0, 2, 3, 4, 1)

# Use grid_sample to deform the volume
deformed_volume = F.grid_sample(volume, deformed_grid, align_corners=False)

# Check the difference between the deformed and original volume
difference = (volume - deformed_volume).sum()

# Should be 0 for zero displacement
print(f"Maximum difference: {difference.item()}")
