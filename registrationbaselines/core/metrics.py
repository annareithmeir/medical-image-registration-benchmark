from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from scipy.spatial.distance import dice
from scipy.spatial import KDTree
import SimpleITK as sitk
import torch

from registrationbaselines.core import utils
from registrationbaselines.core.types import floatArray3Dor4D, floatArray2Dor3D


def jacobian_determinant_from_displacement(displacement: floatArray3Dor4D) -> floatArray2Dor3D:

    displacement = displacement.squeeze()

    assert displacement.ndim == 4 and displacement.shape[-1] == 3 or \
        displacement.ndim == 3 and displacement.shape[-1] == 2, \
        "Displacement field should have shape (h, w, d, 3) or (w, d, 2)"

    if displacement.min() >= -1 or displacement.max() <= 1:
        for dim in range(displacement.shape[-1]):
            displacement[..., dim] = float(
                displacement.shape[-dim - 2] - 1) * displacement[..., dim] / 2.0

    displacement_image = sitk.GetImageFromArray(displacement, isVector=True)
    jacobian_determinant_image = sitk.DisplacementFieldJacobianDeterminant(
        displacement_image)

    return sitk.GetArrayFromImage(jacobian_determinant_image)


def displacement_field_metrics(displacement: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate the fraction of foldings and the standard deviation of the logarithm of the Jacobian determinant.
    """

    epsilon = 1e-6  # so we don't get log(0)

    jacobian_determinant = jacobian_determinant_from_displacement(
        displacement.detach().cpu().numpy())

    # foldings are where the jacobian determinant is negative
    num_foldings = int((jacobian_determinant < 0).astype(float).sum())
    fraction_foldings = num_foldings / jacobian_determinant.size

    # we now add the absolute value of the minimum value of the jacobian determinant to avoid logs of negative values
    # and we add epsilon to avoid log(0)
    log_input = jacobian_determinant + np.abs(np.min(jacobian_determinant))
    log_input += epsilon
    sd_log_det = np.log(log_input).std()

    return sd_log_det, fraction_foldings


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


def dice_score(image1: torch.Tensor, image2: torch.Tensor) -> List[float]:
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

    scores: List[float] = []

    for c in classes:
        # Create binary masks for the current class
        mask1 = (data1 == c).to(torch.uint8).ravel().detach().cpu().numpy()
        mask2 = (data2 == c).to(torch.uint8).ravel().detach().cpu().numpy()

        # Calculate the Dice score using scipy's dice function
        scores.append(1 - dice(mask1, mask2))

    return scores


def hausdorff_distance(image1: torch.Tensor, image2: torch.Tensor, percentile: Optional[float] = None) -> List[float]:
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

    classes = classes.detach().cpu().numpy()
    data1 = data1.detach().cpu().numpy()
    data2 = data2.detach().cpu().numpy()

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


def tre(keypoints_fixed: floatArray2Dor3D,
        keypoints_moving: floatArray2Dor3D,
        keypoints_moving_warped: floatArray2Dor3D,
        spacing_moving: list[float],
        percentile: Optional[float] = None) -> float:
    """
    Calculate the Target Registration Error (TRE) between two sets of keypoints.

    @param keypoints_fixed: fixed keypoints.

    @param keypoints_moving: moving keypoints.

    @param keypoints_moving_warped: warped keypoints.

    @param spacing_moving: The spacing of the moving image.

    @param percentile: Percentile to compute if specified.

    @return: The mean TRE.
    """
    # Calculate the TRE
    all_errors = np.linalg.norm(
        (keypoints_moving_warped - keypoints_fixed) * spacing_moving, axis=1)
    # original TRE
    ori_tre = np.linalg.norm(
        (keypoints_moving - keypoints_fixed) * spacing_moving, axis=1).mean()

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


def read_lanmdarks(keypoints_fixed_path: Path, keypoints_moving_path: Path) -> Tuple[floatArray2Dor3D, floatArray2Dor3D]:
    # hacky but if both paths are the same we are dealign iwth 2d keypoints stored in one file

    # assert that all paths are valid
    assert keypoints_fixed_path.exists(
    ), f"{keypoints_fixed_path} does not exist"
    assert keypoints_moving_path.exists(
    ), f"{keypoints_moving_path} does not exist"

    if keypoints_fixed_path != keypoints_moving_path:
        fixed = np.genfromtxt(keypoints_fixed_path, delimiter=',')
        moving = np.genfromtxt(keypoints_moving_path, delimiter=',')
    else:
        values = np.genfromtxt(keypoints_fixed_path)

        fixed = values[:, 0:2]
        moving = values[:, 2:4]

    assert moving.shape == fixed.shape, \
        "Fixed and moving keypoints should have the same shape"
    assert fixed.shape[-1] == 3 or fixed.shape[-1] == 2, \
        "keypoints should have shape (N,3) or (N,2)"

    return fixed, moving
