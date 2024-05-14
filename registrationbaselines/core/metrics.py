from typing import Dict

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
import nibabel as nib


def dice_score(nifti1_path: str, nifti2_path: str) -> Dict[int, float]:
    """
    Calculate the Dice score between two NIfTI files for each class using scipy's dice function.

    The function reads two NIfTI files, identifies the unique classes in both images, 
    ensures the classes are the same in both images, and calculates the Dice score for each class. 
    The Dice score is a measure of overlap between two samples, defined as:

        Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        nifti1_path (str): Path to the first NIfTI file.
        nifti2_path (str): Path to the second NIfTI file.

    Returns:
        Dict[int, float]: A dictionary where keys are class labels and values are the corresponding Dice scores.
    """

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path).get_fdata()
    nifti2 = nib.load(nifti2_path).get_fdata()

    # Ensure the shapes match
    if nifti1.shape != nifti2.shape:
        raise ValueError("The two NIfTI files must have the same shape.")

    # Find unique classes in the images
    classes1 = np.unique(nifti1)
    classes2 = np.unique(nifti2)

    # Ensure both files have the same classes
    if not np.array_equal(classes1, classes2):
        raise ValueError("The two NIfTI files must have the same classes.")

    dice_scores = {}

    for cls in classes1:
        # Create binary masks for the current class
        mask1 = (nifti1 == cls).astype(int).ravel()
        mask2 = (nifti2 == cls).astype(int).ravel()

        # Calculate the Dice score using scipy's dice function
        current_dice_score = 1 - dice(mask1, mask2)

        dice_scores[int(cls)] = current_dice_score

    return dice_scores
