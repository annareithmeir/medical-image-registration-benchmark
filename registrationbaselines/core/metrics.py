from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
from scipy.spatial import KDTree
import nibabel as nib
import SimpleITK as sitk
from registrationbaselines.core import utils_metrics


def jacobian_determinant_from_displacement(displacement: np.ndarray) -> np.ndarray:
    displacement_image = sitk.GetImageFromArray(
        displacement.squeeze(), isVector=True)
    jacobian_determinant_image = sitk.DisplacementFieldJacobianDeterminant(
        displacement_image)
    return sitk.GetArrayFromImage(jacobian_determinant_image)


def displacement_field_metrics(path_displacement: Path) -> Tuple[float, float]:
    """
    Calculate the fraction of foldings and the standard deviation of the logarithm of the Jacobian determinant.
    """

    epsilon = 1e-6  # so we don't get log(0)

    displacement = nib.load(path_displacement.as_posix()).get_fdata()

    jacobian_determinant = jacobian_determinant_from_displacement(displacement)

    # foldings are where the jacobian determinant is negative
    num_foldings = int((jacobian_determinant < 0).astype(float).sum())
    fraction_foldings = num_foldings / jacobian_determinant.size

    # we now add the absolute value of the minimum value of the jacobian determinant to avoid logs of negative values
    # and we add epsilon to avoid log(0)
    log_input = jacobian_determinant + np.abs(np.min(jacobian_determinant))
    log_input += epsilon
    sd_log_det = np.log(log_input).std()

    return sd_log_det, fraction_foldings


def dice_score(nifti1_path: Path, nifti2_path: Path) -> List[float]:
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
    nifti1 = nib.load(nifti1_path)
    nifti2 = nib.load(nifti2_path)

    # round each value to nearest integer
    data1 = np.round(nifti1.get_fdata()).astype(np.uint8)
    data2 = np.round(nifti2.get_fdata()).astype(np.uint8)

    # Ensure the shapes match
    if data1.shape != data2.shape:
        raise ValueError("The two NIfTI files must have the same shape.")

    # Find unique classes in the images
    classes1 = np.unique(data1)
    classes2 = np.unique(data2)

    assert np.array_equal(classes1,
                          classes2), "Both images should have the same classes."

    # find index of class 0
    idx = np.where(classes1 == 0)

    # remove class 0
    classes1 = np.delete(classes1, idx)
    classes2 = np.delete(classes2, idx)

    scores = []

    for c in classes1:
        # Create binary masks for the current class
        mask1 = (data1 == c).astype(int).ravel()
        mask2 = (data2 == c).astype(int).ravel()

        # Calculate the Dice score using scipy's dice function
        scores.append(1 - dice(mask1, mask2))

    return scores


def hausdorff_distance(nifti1_path: Path, nifti2_path: Path, percentile: Optional[float] = None) -> List[float]:
    """
    Calculate the 95th percentile of the Hausdorff distance between two NIfTI files for each class.

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        float: The 95th percentile of the Hausdorff distances.
    """

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path)
    nifti2 = nib.load(nifti2_path)

    # round each value to nearest integer
    data1 = np.round(nifti1.get_fdata()).astype(np.uint8)
    data2 = np.round(nifti2.get_fdata()).astype(np.uint8)

    # Ensure the shapes match
    if data1.shape != data2.shape:
        raise ValueError("The two NIfTI files must have the same shape.")

    # Find unique classes in the images
    classes1 = np.unique(data1)
    classes2 = np.unique(data2)

    assert np.array_equal(classes1,
                          classes2), "Both images should have the same classes."

    # find index of class 0
    idx = np.where(classes1 == 0)

    # remove class 0
    classes1 = np.delete(classes1, idx)
    classes2 = np.delete(classes2, idx)

    scores = []

    for c in classes1:
        # Get the coordinates of the current class in both images
        coords1 = np.column_stack(np.where(data1 == c))
        coords2 = np.column_stack(np.where(data2 == c))

        if coords1.size == 0 or coords2.size == 0:
            scores.append(np.inf)
        else:
            # Create KD-trees for fast nearest-neighbor lookup
            kdtree1 = KDTree(coords1)
            kdtree2 = KDTree(coords2)

            # Calculate all directed distances from coords1 to coords2
            dists1, _ = kdtree1.query(coords2)
            dists2, _ = kdtree2.query(coords1)

            # Combine both distances
            combined_dists = np.concatenate([dists1, dists2])

            if percentile is not None:
                # Calculate the 95th percentile
                scores.append(np.percentile(combined_dists, 95))
            else:
                # Calculate the maximum distance
                scores.append(np.max(combined_dists))

    return scores


def tre(landmarks_fixed_path: Path,
        landmarks_moving_path: Path,
        displacement_path: Path,
        spacing_moving: list[float],
        percentile: Optional[float] = None) -> float:
    """
    Calculate the Target Registration Error (TRE) between two sets of landmarks.

    Args:
        landmarks_fixed (np.ndarray): The fixed landmarks. landmarks in shape (N,3)
        landmarks_moving (np.ndarray): The moving landmarks. landmarks in shape (N,3)
        displacement (np.ndarray): The displacement field. displacement in shape (h,w,d,[1], 3)
        spacing_moving (Tuple[float, float, float]): The spacing of the moving image.

    Returns:
        float: The mean TRE.
    """

    displacement = nib.load(displacement_path.as_posix()).get_fdata().squeeze()
    landmarks_moving = np.genfromtxt(landmarks_moving_path, delimiter=',')
    landmarks_fixed = np.genfromtxt(landmarks_fixed_path, delimiter=',')
    assert landmarks_moving.shape == landmarks_fixed.shape
    assert landmarks_fixed.shape[-1] == 3

    mov_lms_warped = utils_metrics.deform_landmarks(
        landmarks_moving, displacement)

    # Calculate the TRE
    all_errors = np.linalg.norm((mov_lms_warped - landmarks_fixed)
                                * spacing_moving, axis=1)

    if percentile is not None:
        result = np.percentile(all_errors, percentile)
    else:
        result = all_errors.mean()

    return result
