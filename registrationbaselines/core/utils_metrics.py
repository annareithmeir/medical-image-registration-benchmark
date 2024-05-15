from pathlib import Path

from typing import Dict

import numpy as np
import nibabel as nib


def get_maks_and_classes(nifti1_path: Path, nifti2_path: Path, remove_zero_class: bool = False) -> Dict[int, float]:
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

    # Find unique classes in the images
    classes1 = np.unique(nifti1)
    classes2 = np.unique(nifti2)

    if remove_zero_class:
        if classes1[0] == 0.0:
            classes1 = classes1[1:]
        if classes2[0] == 0.0:
            classes2 = classes2[1:]

    # Ensure both files have the same classes
    if not np.array_equal(classes1, classes2):
        raise ValueError("The two NIfTI files must have the same classes.")

    return nifti1, nifti2, classes1
