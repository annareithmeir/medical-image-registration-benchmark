from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
from scipy.spatial import KDTree
import nibabel as nib
import SimpleITK as sitk
from registrationbaselines.core import utils


def jacobian_determinant_from_displacement(displacement: np.ndarray) -> np.ndarray:

    displacement = displacement.squeeze()

    assert displacement.ndim == 4 and displacement.shape[-1] == 3 or \
        displacement.ndim == 3 and displacement.shape[-1] == 2, \
        "Displacement field should have shape (h, w, d, 3) or (w, d, 2)"

    if displacement.shape[-1] != 3 and displacement.shape[0] == 3:
        displacement = displacement.transpose(1, 2, 3, 0)

    displacement_image = sitk.GetImageFromArray(displacement, isVector=True)
    jacobian_determinant_image = sitk.DisplacementFieldJacobianDeterminant(
        displacement_image)
    return sitk.GetArrayFromImage(jacobian_determinant_image)


def displacement_field_metrics(displacement: sitk.Image) -> Tuple[float, float]:
    """
    Calculate the fraction of foldings and the standard deviation of the logarithm of the Jacobian determinant.
    """

    epsilon = 1e-6  # so we don't get log(0)

    jacobian_determinant = jacobian_determinant_from_displacement(
        sitk.GetArrayFromImage(displacement))

    # foldings are where the jacobian determinant is negative
    num_foldings = int((jacobian_determinant < 0).astype(float).sum())
    fraction_foldings = num_foldings / jacobian_determinant.size

    # we now add the absolute value of the minimum value of the jacobian determinant to avoid logs of negative values
    # and we add epsilon to avoid log(0)
    log_input = jacobian_determinant + np.abs(np.min(jacobian_determinant))
    log_input += epsilon
    sd_log_det = np.log(log_input).std()

    return sd_log_det, fraction_foldings


def preprocess_segmentations(image1: sitk.Image, image2: sitk.Image):
    """
    Prepares egmentations for evaluation by ensuring the classes are the same and removing class 0.
    """
    data1 = sitk.GetArrayFromImage(image1)
    data2 = sitk.GetArrayFromImage(image2)

    # round each value to nearest integer
    data1 = np.round(data1).astype(np.uint8)
    data2 = np.round(data2).astype(np.uint8)

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

    return classes1, data1, data2


def dice_score(image1: sitk.Image, image2: sitk.Image) -> List[float]:
    """
    Calculate the Dice score between two NIfTI files using scipy's dice function. It is assumed that both
    images have only one class.

    The function reads two NIfTI files, ensures the classes are the same in both images, and calculates
    the Dice score for each class. The Dice score is a measure of overlap between two samples, defined as:

        Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)

    Returns:
        float: The Dice score between the two NIfTI files.
    """

    classes, data1, data2 = preprocess_segmentations(image1, image2)

    scores = []

    for c in classes:
        # Create binary masks for the current class
        mask1 = (data1 == c).astype(int).ravel()
        mask2 = (data2 == c).astype(int).ravel()

        # Calculate the Dice score using scipy's dice function
        scores.append(1 - dice(mask1, mask2))

    return scores


def hausdorff_distance(image1: sitk.Image, image2: sitk.Image, percentile: Optional[float] = None) -> List[float]:
    """
    Calculate the 95th percentile of the Hausdorff distance between two NIfTI files for each class.

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        float: The 95th percentile of the Hausdorff distances.
    """

    classes, data1, data2 = preprocess_segmentations(image1, image2)

    scores = []

    for c in classes:
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
    landmarks_fixed, landmarks_moving = read_lanmdarks(
        landmarks_fixed_path, landmarks_moving_path)

    assert landmarks_moving.shape == landmarks_fixed.shape
    assert landmarks_fixed.shape[-1] == 3 or landmarks_fixed.shape[-1] == 2

    mov_lms_warped = utils.deform_landmarks(
        landmarks_moving, displacement)

    # Calculate the TRE
    all_errors = np.linalg.norm((mov_lms_warped - landmarks_fixed)
                                * spacing_moving, axis=1)
    # original TRE
    ori_tre = np.linalg.norm(
        (landmarks_moving - landmarks_fixed) * spacing_moving, axis=1).mean()

    print("\n")
    if all_errors.mean() < ori_tre:
        print(
            f"TRE is smaller than original TRE by % {100*(ori_tre - all_errors.mean())/ori_tre:2f}.\nFrom {ori_tre} to {all_errors.mean()}")
    else:
        print(
            f"TRE is larger than original TRE by % {100*(all_errors.mean() - ori_tre)/ori_tre:2f}. \nFrom {ori_tre} to {all_errors.mean()}")

    if percentile is not None:
        result = np.percentile(all_errors, percentile)
    else:
        result = all_errors.mean()

    return result


def read_lanmdarks(landmarks_fixed_path: Path, landmarks_moving_path: Path) -> np.ndarray:
    # hacky but if both paths are the same we are dealign iwth 2d landmarks stored in one file
    if landmarks_fixed_path != landmarks_moving_path:
        return np.genfromtxt(landmarks_fixed_path, delimiter=','), np.genfromtxt(landmarks_moving_path, delimiter=',')

    else:
        values = np.genfromtxt(landmarks_fixed_path)

        fixed = values[:, 0:2]
        moving = values[:, 2:4]

        return fixed, moving
