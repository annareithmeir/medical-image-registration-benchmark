import numpy as np
from scipy.ndimage import binary_erosion


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
