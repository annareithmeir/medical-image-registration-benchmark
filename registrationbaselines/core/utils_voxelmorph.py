
from typing import List

import torch
import torch.nn.functional as F


def get_new_voxelmorph_image_shape(old_shape: List[int],
                                   number_of_layers_in_encoder: int) -> List[int]:
    """
    @brief Adjusts the input shape to be compatible with Voxelmorph requirements.

    This function calculates a new shape for the input data such that each dimension is 
    a multiple of 2^n, where n is the number of layers in the encoder of the Voxelmorph model.
    The function ensures that the data is padded and not cropped, so no information is lost.

    @param old_shape The original shape of the input data (List of 3 integers).
    @param number_of_layers_in_encoder The number of layers in the encoder of the model.

    @return A new shape (List of 3 integers) where each dimension is a multiple of 2^n.
    """

    new_shape: List[int] = [0, 0, 0]

    for i in range(3):
        new_shape[i] = torch.ceil(torch.tensor(
            old_shape[i] / (2 ** number_of_layers_in_encoder))).int().item() * (2 ** number_of_layers_in_encoder)

    return new_shape


def pad_tensor_to_shape(tensor: torch.Tensor,
                        new_shape: List[int]) -> torch.Tensor:
    """
    @brief Pad a tensor at the end of each dimension to the new shape

    @param tensor The tensor which will be padded
    @param new_shape The shape that the resuling tensor will be padded to

    @return The padded tensor
    """

    shape = tensor.shape[2:]
    pad: List[int] = [0, 0, 0]

    for i in range(3):
        size = shape[i]
        pad[i] = new_shape[i] - size  # Only pad at the end

    # Reverse pad list and interleave with zeros for F.pad format
    pad = [item for sublist in zip([0]*3, reversed(pad))
           for item in sublist]

    if pad == [0, 0, 0, 0, 0, 0]:
        return tensor

    padded = F.pad(tensor, pad)

    return padded


def crop_tensor_to_shape(tensor: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """
    Crops the tensor to the specified shape, regardless of the number of dimensions.

    @param tensor (torch.Tensor): The tensor to be cropped.
    @param shape (List[int]): The desired shape to crop to. It should have the same length as the number of dimensions in the tensor.

    @return torch.Tensor: The cropped tensor with the specified shape.
    """

    if len(tensor.shape) != len(shape):
        raise ValueError(
            "The shape list must have the same number of dimensions as the tensor.")

    # Dynamically create the slicing for each dimension
    slices = tuple(slice(0, dim) for dim in shape)

    # Apply the slices to crop the tensor
    tensor_cropped = tensor[slices]

    return tensor_cropped
