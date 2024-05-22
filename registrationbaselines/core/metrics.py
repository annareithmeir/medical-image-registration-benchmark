from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates
import nibabel as nib
import SimpleITK as sitk


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

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path).get_fdata()
    nifti2 = nib.load(nifti2_path).get_fdata()

    assert np.array_equal(np.unique(nifti1), np.unique(
        nifti2)), "Both images should have the same classes."
    assert len(np.unique(nifti1)) == 2, "Both images should have only one class."
    assert np.unique(nifti1)[0] == 0 and np.unique(nifti1)[
        1] == 1, "Both images should have only one class."

    # Get the coordinates of the current class in both images
    coords1 = np.column_stack(np.where(nifti1 == 1))
    coords2 = np.column_stack(np.where(nifti2 == 1))

    if coords1.size == 0 or coords2.size == 0:
        current_hausdorff_distance = np.inf
    else:
        # Calculate the Hausdorff distance using scipy's directed_hausdorff function
        hd1 = directed_hausdorff(coords1, coords2)[0]
        hd2 = directed_hausdorff(coords2, coords1)[0]
        current_hausdorff_distance = max(hd1, hd2)

    return current_hausdorff_distance


def hausdorff95_distance(nifti1_path: Path, nifti2_path: Path) -> float:
    """
    Calculate the 95th percentile of the Hausdorff distance between two NIfTI files for each class.

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        float: The 95th percentile of the Hausdorff distances.
    """

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path).get_fdata()
    nifti2 = nib.load(nifti2_path).get_fdata()

    assert np.array_equal(np.unique(nifti1), np.unique(
        nifti2)), "Both images should have the same classes."
    assert len(np.unique(nifti1)) == 2, "Both images should have only one class."
    assert np.unique(nifti1)[0] == 0 and np.unique(nifti1)[
        1] == 1, "Both images should have only one class."

    # Get the coordinates of the current class in both images
    coords1 = np.column_stack(np.where(nifti1 == 1))
    coords2 = np.column_stack(np.where(nifti2 == 1))

    if coords1.size == 0 or coords2.size == 0:
        hausdorff95 = np.inf
    else:
        # Create KD-trees for fast nearest-neighbor lookup
        kdtree1 = KDTree(coords1)
        kdtree2 = KDTree(coords2)

        # Calculate all directed distances from coords1 to coords2
        dists1, _ = kdtree1.query(coords2)
        dists2, _ = kdtree2.query(coords1)

        # Combine both distances
        combined_dists = np.concatenate([dists1, dists2])

        # Calculate the 95th percentile
        hausdorff95 = np.percentile(combined_dists, 95)

    return hausdorff95


def tre(landmarks_fixed: np.ndarray,
        landmarks_moving: np.ndarray,
        displacement: np.ndarray,
        spacing_moving: Tuple[float, float],
        percentile: Optional[float] = None) -> Tuple[float, list]:
    """
    Calculate the Target Registration Error (TRE) between two sets of landmarks.

    Args:
        landmarks_fixed (np.ndarray): The fixed landmarks.
        landmarks_moving (np.ndarray): The moving landmarks.
        displacement (np.ndarray): The displacement field.
        spacing_moving (Tuple[float, float, float]): The spacing of the moving image.

    Returns:
        Tuple[float, list]: The mean TRE and a list of detailed TRE values.
    """

    displacement = displacement.squeeze()

    # Map the moving landmarks to the fixed landmarks using the displacement field
    mov_lms_disp_x = map_coordinates(
        displacement[:, :, :, 0], landmarks_moving.transpose())
    mov_lms_disp_y = map_coordinates(
        displacement[:, :, :, 1], landmarks_moving.transpose())
    mov_lms_disp_z = map_coordinates(
        displacement[:, :, :, 2], landmarks_moving.transpose())
    mov_lms_disp = np.array(
        (mov_lms_disp_x, mov_lms_disp_y, mov_lms_disp_z)).transpose()

    # Calculate the warped moving landmarks
    mov_lms_warped = landmarks_moving + mov_lms_disp

    # Calculate the TRE
    all_errors = np.linalg.norm((mov_lms_warped - landmarks_fixed)
                                * spacing_moving, axis=1)

    if percentile is not None:
        result = np.percentile(all_errors, percentile)
    else:
        result = all_errors.mean()

    return result
