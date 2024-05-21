from pathlib import Path

from typing import Dict

import numpy as np
import nibabel as nib


def extract_class(segmentation: np.ndarray, class_value: int) -> np.ndarray:
    """
    Extract a specific class from the segmentation.

    Args:
    segmentation (np.ndarray): The segmentation data.
    class_value (int): The class value to extract.

    Returns:
    np.ndarray: A binary mask where the class value is set to 1 and others to 0.
    """
    assert isinstance(
        segmentation, np.ndarray), "Segmentation must be a numpy array."
    assert isinstance(class_value, np.uint8), "Class value must be an integer."

    return (segmentation == class_value).astype(np.uint8)


    """
    Load two NIfTI files and return them and unique classes - class 0 is removed.

    params:
    nifti1_path (Path): Path to the first NIfTI file.
    nifti2_path (Path): Path to the second NIfTI file.
    remove_zero_class (bool): Whether to remove the zero class from the images (background class).
    """

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path.as_posix()).get_fdata()
    nifti2 = nib.load(nifti2_path.as_posix()).get_fdata()

    # Ensure the shapes match
    if nifti1.shape != nifti2.shape:
        raise ValueError("The two NIfTI files must have the same shape.")

    # due to interpolation, the values might ot be exactly 0 or 1
    nifti1 = np.ceil(nifti1)
    nifti2 = np.ceil(nifti2)

    # Find unique classes in the images
    classes1 = np.unique(nifti1)
    classes2 = np.unique(nifti2)

    if classes1[0] == 0.0:
        classes1 = classes1[1:]
    if classes2[0] == 0.0:
        classes2 = classes2[1:]

    # Ensure both files have the same classes
    if not np.array_equal(classes1, classes2):
        raise ValueError("The two NIfTI files must have the same classes.")

    return nifti1, nifti2, classes1
