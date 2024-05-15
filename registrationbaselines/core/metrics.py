from pathlib import Path
import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
import nibabel as nib

from registrationbaselines.core import utils_metrics


def sdlogj(path_displacement: Path) -> Tuple[float, float]:
    """
    Calculate the number of foldings and the standard deviation of the logarithm of the Jacobian determinant.
    """

    displacement = nib.load(path_displacement.as_posix()).get_fdata()

    warnings.warn("I don't know what the +3 and clip does here.")

    temp = utils_metrics.jacobian_determinant(
        displacement.transpose((3, 4, 0, 1, 2)))

    determinant = (temp + 3).clip(0.000000001, 1000000000)

    num_foldings = (determinant <= 0).astype(float).sum()

    sd_log_det = np.log(determinant).std()

    return sd_log_det, num_foldings


def dice_score(nifti1_path: Path, nifti2_path: Path) -> Dict[int, float]:
    """
    Calculate the Dice score between two NIfTI files for each class using scipy's dice function.

    The function reads two NIfTI files, identifies the unique classes in both images, 
    ensures the classes are the same in both images, and calculates the Dice score for each class. 
    The Dice score is a measure of overlap between two samples, defined as:

        Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        Dict[int, float]: A dictionary where keys are class labels and values are the corresponding Dice scores.
    """

    nifti1, nifti2, classes1 = utils_metrics.get_maks_and_classes(
        nifti1_path, nifti2_path)

    dice_scores = {}

    for cls in classes1:
        # Create binary masks for the current class
        mask1 = (nifti1 == cls).astype(int).ravel()
        mask2 = (nifti2 == cls).astype(int).ravel()

        # Calculate the Dice score using scipy's dice function
        current_dice_score = 1 - dice(mask1, mask2)

        dice_scores[f"dice_{int(cls)}"] = current_dice_score

    return dice_scores


def hausdorff_distance(nifti1_path: Path, nifti2_path: Path) -> Dict[int, float]:
    """
    Calculate the Hausdorff distance between two NIfTI files for each class.

    The function reads two NIfTI files, identifies the unique classes in both images,
    ensures the classes are the same in both images, and calculates the Hausdorff distance for each class.
    The Hausdorff distance is defined as the maximum distance of a set to the nearest point in the other set.

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        Dict[int, float]: A dictionary where keys are class labels and values are the corresponding Hausdorff distances.
    """

    nifti1, nifti2, classes1 = utils_metrics.get_maks_and_classes(
        nifti1_path, nifti2_path)

    hausdorff_distances = {}

    for cls in classes1:
        # Get the coordinates of the current class in both images
        coords1 = np.column_stack(np.where(nifti1 == cls))
        coords2 = np.column_stack(np.where(nifti2 == cls))

        if coords1.size == 0 or coords2.size == 0:
            current_hausdorff_distance = np.inf
        else:
            # Calculate the Hausdorff distance using scipy's directed_hausdorff function
            hd1 = directed_hausdorff(coords1, coords2)[0]
            hd2 = directed_hausdorff(coords2, coords1)[0]
            current_hausdorff_distance = max(hd1, hd2)

        hausdorff_distances[int(cls)] = current_hausdorff_distance

    return hausdorff_distances
