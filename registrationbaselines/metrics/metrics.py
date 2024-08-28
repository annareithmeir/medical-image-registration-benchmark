from pathlib import Path
from math import nan

from typing import Tuple, Optional, List, Dict

import numpy as np
import scipy
import scipy.ndimage
import SimpleITK as sitk
import torch
import monai

from registrationbaselines.metrics import hd95, utils_metrics
from registrationbaselines.core.types import floatArray3Dor4D, floatArray2Dor3D


def get_non_zero_unique_classes(image1: torch.Tensor, image2: torch.Tensor) -> List[int]:
    """
    Returns a sorted list of uniqe classes (without class 0)
    """

    # get the unique classes
    unique_classes = torch.unique(torch.cat((image1, image2))).tolist()

    # remove class 0
    if 0 in unique_classes:
        unique_classes.remove(0)

    return sorted(unique_classes)


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


def jacobian_determinant_from_displacement_monai(displacement: torch.Tensor) -> torch.Tensor:
    displacement = displacement.permute(3, 1, 2, 0)
    jacobian_determinant = monai.losses.compute_jacobian_determinant(
        displacement)
    return jacobian_determinant


def jacobian_determinant_from_displacement_l2r(disp: torch.Tensor) -> torch.Tensor:

    disp = disp.permute(3, 0, 1, 2)
    disp = disp.unsqueeze(0)

    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(
                               disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(
                               disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(
                               disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
        jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :,
                                                                                               :]) - \
        jacobian[1, 0, :, :, :] * (
        jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2,
                                                                                               1, :, :,
                                                                                               :]) + \
        jacobian[2, 0, :, :, :] * (
        jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1,
                                                                                               1, :, :, :])

    return jacdet


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


def displacement_field_metrics_monai(displacement: torch.Tensor) -> Tuple[float, float]:
    pass


def displacement_field_metrics_l2r(displacement: torch.Tensor) -> Tuple[float, float]:
    pass


def is_class_present_in_only_one(array1: np.ndarray[bool],
                                 array2: np.ndarray[bool]) -> bool:
    """
    Returns true if one image has only False and the other not
    """

    all_false_array1 = np.all(array1 == False)
    all_false_array2 = np.all(array2 == False)

    if all_false_array1 and not all_false_array2:
        return True
    elif not all_false_array1 and all_false_array2:
        return True
    else:
        return False


def dice_score(image1: torch.Tensor, image2: torch.Tensor) -> Dict[str, float]:
    """
    Calculate the Dice score between two torch Tensors using scipy's dice function.
    Supports multi class.
    If a class is only present in one image, a score of nan is appended


    The Dice score is a measure of overlap between two samples, defined as:

        Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)

    @param image1: The first image.
    @param image2: The second image.
    @return: A dictionary containing the Dice score for each class.
    """

    if image1.dtype != torch.uint8 or image2.dtype != torch.uint8:
        raise ValueError("Both images should be of type uint8")
    if image1.shape != image2.shape:
        raise ValueError("Both images should have the same shape")

    unique_classes = get_non_zero_unique_classes(image1, image2)

    scores: Dict[str, float] = {}

    for cls in unique_classes:
        # Create binary masks for the current class
        mask1 = (image1 == cls).ravel().detach().cpu().numpy()
        mask2 = (image2 == cls).ravel().detach().cpu().numpy()

        # if the class is only in one image, set the score to -1
        if is_class_present_in_only_one(mask1, mask2):
            scores[str(cls)] = nan

        else:
            scores[str(cls)] = 1.0 - scipy.spatial.distance.dice(mask1, mask2)

    return scores


def dice_score_monai(image1: torch.Tensor, image2: torch.Tensor) -> List[float]:
    """

    @param image1: (h,d,w)
    @param image2: (h,d,w)
    @return:
    """

    # Find the union of classes in both label maps
    unique_classes = torch.unique(torch.cat((image1, image2)))

    # Map the labels to continuous indices
    class_to_index = {cls.item(): i for i, cls in enumerate(unique_classes)}
    index_to_class = {i: cls.item() for i, cls in enumerate(unique_classes)}

    image1_mapped = image1.clone()
    image2_mapped = image2.clone()

    # Re-map label maps to a common set of classes
    # for original_class, new_index in class_to_index.items():
    #     image1_mapped[image1 == original_class] = new_index
    #     image2_mapped[image2 == original_class] = new_index

    image2_mapped = image2
    image1_mapped = image1

    # Number of classes after mapping
    # num_classes = len(unique_classes)
    num_classes = -1

    one_hot1 = torch.nn.functional.one_hot(image1_mapped.unsqueeze(
        0).unsqueeze(0), num_classes=num_classes).transpose(-1, 1).squeeze(-1)
    one_hot2 = torch.nn.functional.one_hot(image2_mapped.unsqueeze(
        0).unsqueeze(0), num_classes=num_classes).transpose(-1, 1).squeeze(-1)

    dice_metric = monai.metrics.DiceMetric(
        include_background=False, reduction="none", get_not_nans=False)
    dice_score = dice_metric(y_pred=one_hot1, y=one_hot2)

    dice_score = dice_score.squeeze().tolist()
    if type(dice_score) == float:
        dice_score = [dice_score]
    return dice_score


def dice_score_monai_new(image1: torch.Tensor, image2: torch.Tensor) -> List[float]:

    def dice_score_single(image1: torch.Tensor, image2: torch.Tensor, class_label: int) -> float:
        image1_class = (image1 == class_label)
        image2_class = (image2 == class_label)

        dice_metric = monai.metrics.DiceMetric(include_background=False,
                                               reduction="none",
                                               get_not_nans=False)
        dice_score = dice_metric(y_pred=image1_class, y=image2_class)

        return dice_score.squeeze().tolist()

    unique_classes = sorted(torch.unique(torch.cat((image1, image2))).tolist())

    # remove zero class
    if unique_classes[0] == 0:
        unique_classes = unique_classes[1:]

    scores = []
    for cls in unique_classes:
        scores.append(dice_score_single(image1, image2, cls))

    return scores


def dice_score_l2r(fixed: torch.Tensor, moving_warped: torch.Tensor, moving: torch.Tensor) -> List[float]:
    def compute_dice_coefficient(mask_gt, mask_pred):
        """Computes soerensen-dice coefficient.

        compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred`.

        Args:
          mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
          mask_pred: 3-dim Numpy array of type bool. The predicted mask.

        Returns:
          the dice coeffcient as float. If both masks are empty, the result is NaN.
        """
        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return 0
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum
    classes = get_non_zero_unique_classes(fixed, moving)
    print(classes)

    dice = []
    for i in classes:
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(float(compute_dice_coefficient(
                (fixed == i), (moving_warped == i))))

    return list(dice)


def hausdorff_distance_learn2reg(image1: torch.Tensor, image2: torch.Tensor, percentile: float) -> List[float]:
    """
    Calculate the 95th percentile of the Hausdorff distance between two NIfTI files for each class.

    Args:
        nifti1_path (Path): Path to the first NIfTI file.
        nifti2_path (Path): Path to the second NIfTI file.

    Returns:
        float: The 95th percentile of the Hausdorff distances.
    """

    classes = get_non_zero_unique_classes(image1, image2)
    # print(classes)

    image1 = image1.detach().cpu().numpy()
    image2 = image2.detach().cpu().numpy()

    hd95_vals = []
    for i in classes:
        if ((image1 == i).sum() == 0) or ((image2 == i).sum() == 0):
            hd95_vals.append(np.NAN)
        else:
            hd95_vals.append(hd95.compute_robust_hausdorff(hd95.compute_surface_distances(
                (image1 == i), (image2 == i), np.ones(3)), percentile))

    return hd95_vals


def hausdorff_distance_monai(image1, image2,  p=95, spacing=None):
    """
    images must be one how and classdim is at dim zero
    @param image1:
    @param image2:
    @param p:
    @param spacing:
    @return:
    """

    assert image2.shape == image1.shape
    image1 = image1.unsqueeze(0)  # add batch dim
    image2 = image2.unsqueeze(0)

    # image1 = torch.movedim(image1, -1, 1)
    # image2 = torch.movedim(image2, -1, 1)

    hd = monai.metrics.compute_hausdorff_distance(
        image1, image2, percentile=p, spacing=spacing)

    return hd.detach().numpy()[0]


def tre(keypoints_fixed: torch.Tensor,
        keypoints_moving: torch.Tensor,
        keypoints_moving_warped: torch.Tensor,
        spacing_moving: Tuple[float, ...],
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
    all_errors = torch.norm(
        (keypoints_moving_warped - keypoints_fixed) * torch.tensor(spacing_moving), dim=1)
    # original TRE
    ori_tre = torch.norm(
        (keypoints_moving - keypoints_fixed) * torch.tensor(spacing_moving), dim=1).mean()

    print("\n")
    if all_errors.mean() < ori_tre:
        print(
            f"TRE is smaller than original TRE by % {100*(ori_tre - all_errors.mean())/ori_tre:2f}.\nFrom {ori_tre} to {all_errors.mean()}")
    else:
        print(
            f"TRE is larger than original TRE by % {100*(all_errors.mean() - ori_tre)/ori_tre:2f}. \nFrom {ori_tre} to {all_errors.mean()}")

    if percentile:
        result = torch.quantile(all_errors, percentile/100)
    else:
        result = all_errors.mean()

    return result.item()
