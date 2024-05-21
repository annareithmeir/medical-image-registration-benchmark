from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
import nibabel as nib
import SimpleITK as sitk

from registrationbaselines.core import utils_metrics


def sdlogj(path_displacement: Path) -> Tuple[float, float]:
    """
    Calculate the number of foldings and the standard deviation of the logarithm of the Jacobian determinant.
    """

    epsilon = 1e-6  # so we don't get log(0)

    displacement = nib.load(path_displacement.as_posix()).get_fdata()

    # create an sitk image (but remove the 1 dimension)
    displacement_image = sitk.GetImageFromArray(
        displacement.squeeze(), isVector=True)
    jacobian_determinant_image = sitk.DisplacementFieldJacobianDeterminant(
        displacement_image)
    jacobian_determinant = sitk.GetArrayFromImage(jacobian_determinant_image)

    # foldings are where the jacobian determinant is negative
    num_foldings = int((jacobian_determinant < 0).astype(float).sum())

    # we now add the absolute value of the minimum value of the jacobian determinant to avoid logs of negative values
    # and we add epsilon to avoid log(0)
    log_input = jacobian_determinant + np.abs(np.min(jacobian_determinant))
    log_input += epsilon
    sd_log_det = np.log(log_input).std()

    return sd_log_det, num_foldings


def dice_score(nifti1_path: Path, nifti2_path: Path) -> float:
    """
    Calculate the Dice score between two NIfTI files using scipy's dice function. It is assumed that both
    images have only one class.

    The function reads two NIfTI files, ensures the classes are the same in both images, and calculates
    the Dice score for each class. The Dice score is a measure of overlap between two samples, defined as:

        Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        float: The Dice score between the two NIfTI files.
    """

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path).get_fdata()
    nifti2 = nib.load(nifti2_path).get_fdata()

    assert np.array_equal(np.unique(nifti1), np.unique(
        nifti2)), "Both images should have the same classes."
    assert len(np.unique(nifti1)) == 2, "Both images should have only one class."
    assert np.unique(nifti1)[0] == 0 and np.unique(nifti1)[
        1] == 1, "Both images should have only one class."

    # Create binary masks for the current class
    mask1 = (nifti1 == 1).astype(int).ravel()
    mask2 = (nifti2 == 1).astype(int).ravel()

    # Calculate the Dice score using scipy's dice function
    score = 1 - dice(mask1, mask2)

    return score


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
