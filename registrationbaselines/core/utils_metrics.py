from typing import List, Union

import torch


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
