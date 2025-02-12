from typing import List, Tuple

import torch
import torch.nn.functional as F


def pad_tensor_to_shape(tensor: torch.Tensor,
                        new_shape: List[int]) -> torch.Tensor:
    """
    @brief Pad a tensor at the end of each dimension to the new shape

    @param tensor The tensor which will be padded
    @param new_shape The shape that the resulting tensor will be padded to

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


def get_new_lapirn_image_shape(old_shape: List[int]) -> List[int]:
    """
    @brief Adjusts the input shape to be compatible with LapIRN requirements.

    This function calculates a new shape for the input data such that each dimension is
    a multiple of 8.
    The function ensures that the data is padded and not cropped, so no information is lost.

    @param old_shape The original shape of the input data (List of 3 integers).

    @return A new shape (List of 3 integers) where each dimension is a multiple of 8.
    """

    new_shape: List[int] = [0, 0, 0]

    for i in range(3):
        x = old_shape[i]
        new_shape[i] = x if x % 8 == 0 else x + (8 - x % 8)
    return new_shape


def get_new_voxelmorph_image_shape(old_shape: Tuple[int, ...],
                                   number_of_layers_in_encoder: int) -> Tuple[int, ...]:
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

    return tuple(new_shape)


def reshape_tensor(tensor: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
    """
    Reshape a tensor by padding or cropping each dimension as needed.

    Args:
    tensor (torch.Tensor): The input tensor to reshape
    new_shape (tuple): The desired output shape

    Returns:
    torch.Tensor: The reshaped tensor
    """

    # Ensure the new_shape has the same number of dimensions as the input tensor
    assert len(new_shape) == tensor.dim(
    ), "New shape must have the same number of dimensions as the input tensor"

    current_shape = tensor.shape

    # Initialize the pad and crop parameters
    pad_sizes = []
    crop_slices = []

    for i, (current, new) in enumerate(zip(current_shape, new_shape)):
        if new > current:
            # Padding needed
            pad_left = (new - current) // 2
            pad_right = new - current - pad_left
            pad_sizes.extend([pad_left, pad_right])
        elif new < current:
            # Cropping needed
            crop_start = (current - new) // 2
            crop_end = crop_start + new
            crop_slices.append(slice(crop_start, crop_end))
        else:
            # No change needed
            pad_sizes.extend([0, 0])
            crop_slices.append(slice(None))

    # Reverse pad_sizes because F.pad expects them in reverse order
    pad_sizes.reverse()

    # Pad the tensor if needed
    if any(pad_sizes):
        tensor = F.pad(tensor, pad_sizes)

    # Crop the tensor if needed
    if any(s.start is not None or s.stop is not None for s in crop_slices):
        tensor = tensor[tuple(crop_slices)]

    return tensor
