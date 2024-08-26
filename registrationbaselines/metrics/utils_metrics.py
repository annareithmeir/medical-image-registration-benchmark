from typing import List, Tuple, Union

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


def preprocess_segmentations(image1: torch.Tensor, image2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares egmentations for evaluation by ensuring the classes are the same and removing class 0.
    """

    # round each value to nearest integer
    image1 = torch.round(image1).to(torch.uint8)
    image2 = torch.round(image2).to(torch.uint8)

    # Ensure the shapes match
    if image1.shape != image2.shape:
        raise ValueError("The two NIfTI files must have the same shape.")

    # Find unique classes in the images
    classes1 = torch.unique(image1)
    classes2 = torch.unique(image2)

    # Find common classes
    common_classes = torch.tensor(
        [c for c in classes1 if c in classes2], device=image1.device, dtype=classes1.dtype)

    if not torch.equal(classes1, classes2):
        Warning(
            "Both images should have the same classes. Continuing with classes common for both segmentations.")

    # find index of class 0
    idx = torch.where(classes1 == 0)[0]

    # remove class 0
    mask = torch.ones(len(common_classes), dtype=bool, device=classes1.device)
    mask[idx] = 0
    common_classes = torch.masked_select(common_classes, mask)

    return common_classes, image1, image2
