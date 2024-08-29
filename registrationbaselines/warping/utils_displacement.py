from typing import Union

import torch

from registrationbaselines.core import utils


def compute_grid(image_size: torch.Size,
                 dtype: torch.dtype = torch.float32,
                 device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Compute a normalized grid for a given image size.

    @param image_size: The size of the image. Should be a list of two or three integers.
    @type image_size: List[int]

    @param dtype: The desired data type of the returned grid.
    @type dtype: torch.dtype

    @param device: The desired device of the returned grid.
    @type device: Union[str, torch.device]

    @return: A tensor representing the normalized grid.
    @rtype: torch.Tensor
    """

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        raise ValueError(f"Error: {dim} is not a valid grid dimension.")


def displacement_to_unit_displacement(displacement: torch.Tensor) -> torch.Tensor:
    """
    Convert a displacement field to a unit displacement field.
    The standard unit of displacement is a half-image, so a displacement vector of magnitude 2
    means that the displacement distance is equal to the side length of the displaced image.
    """

    disp = torch.zeros_like(displacement)

    for dim in range(displacement.shape[-1]):
        disp[..., dim] = 2.0 * displacement[..., dim] / \
            float(displacement.shape[-dim - 2] - 1)

    return disp


def unit_displacement_to_displacement(displacement: torch.Tensor) -> torch.Tensor:

    disp = torch.zeros_like(displacement)

    for dim in range(displacement.shape[-1]):
        disp[..., dim] = float(
            displacement.shape[-dim - 2] - 1) * displacement[..., dim] / 2.0

    return disp


def reverse_axis(image: torch.Tensor) -> torch.Tensor:
    """
    Flips the order of the axis representing the space dimensions (preceeding dimensions are ignored).
    Respectively, the axis holding the vectors is flipped as well

    Note: the method is inplace
    """
    # reverse order of axis to follow the convention of SimpleITK
    order = list(reversed(range(image.ndim-1)))
    order.append(len(order))
    image = image.squeeze_().permute(tuple(order))
    image = utils.flip(image, image.ndim-1)

    return image
