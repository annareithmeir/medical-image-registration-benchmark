from warnings import warn

import numpy as np
from scipy.ndimage import binary_erosion
import torch


def multilabel_to_boundary(label_map: np.ndarray):
    label_map = label_map.astype(int)
    num_labels = np.max(label_map)
    boundary_label_map = np.zeros_like(label_map)

    # Iterate over each label
    for label in range(1, num_labels + 1):
        # Create a binary mask for the current label
        label_mask = (label_map == label)

        # Apply binary erosion to the binary mask
        eroded_label_mask = binary_erosion(label_mask)

        # Compute the boundary mask for the current label
        boundary_mask = label_mask.astype(
            np.int8) - eroded_label_mask.astype(np.int8)

        # Assign the boundary mask to the boundary label map
        boundary_label_map += boundary_mask * label
    return boundary_label_map


def check_if_deformed_images_are_similar_enough(image1: torch.Tensor,
                                                image2: torch.Tensor,
                                                threshold: float = 0.001) -> None:
    """
    Check if two images are similar enough, by comparing the sum absolute difference between the images.
    """

    if image1.shape != image2.shape:
        raise ValueError(
            f"The two images must have the same shape, but the provided images have shapes {image1.shape} and {image2.shape}.")

    difference = torch.sum(torch.abs(image1 - image2))

    number_of_voxels = torch.numel(image1)

    ratio: torch.Tensor = difference / number_of_voxels

    if ratio > threshold:
        warn(f"Deformed image from registration and deformed image from deformation field differ by {ratio}.\n \
                        The deformation field is probbaly not saved correctly.")
