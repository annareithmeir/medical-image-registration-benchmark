from pathlib import Path

from typing import Tuple
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


def save_class_nifti(original_img: nib.Nifti1Image,
                     class_mask: np.ndarray,
                     output_path: str) -> None:
    """
    Save a binary mask as a NIfTI file.

    Args:
    original_img (nib.Nifti1Image): The original NIfTI image.
    class_mask (np.ndarray): The mask for the class.
    output_path (str): The output file path.

    Returns:
    None
    """

    # change to binary mask
    class_mask[class_mask > 0] = 1

    class_img = nib.Nifti1Image(class_mask,
                                original_img.affine,
                                original_img.header)

    nib.save(class_img, output_path)


def get_segmentation_classes(nifti_path: Path) -> np.ndarray:
    """
    Load a NIfTI file and return the unique classes in the image.

    params:
    nifti_path (Path): Path to the NIfTI file.
    """

    # Load the NIfTI file
    nifti = nib.load(nifti_path.as_posix()).get_fdata()

    # Find unique classes in the image
    classes = np.unique(nifti)

    if classes[0] == 0.0:
        classes = classes[1:]

    return classes


def get_maks_and_classes(nifti1_path: Path, nifti2_path: Path) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray]:
    """
    Load two NIfTI files and return them and unique classes - class 0 is removed.

    params:
    nifti1_path (Path): Path to the first NIfTI file.
    nifti2_path (Path): Path to the second NIfTI file.
    remove_zero_class (bool): Whether to remove the zero class from the images (background class).
    """

    # Load the NIfTI files
    nifti1 = nib.load(nifti1_path.as_posix())
    nifti2 = nib.load(nifti2_path.as_posix())

    # round each value to nearest integer
    data1 = np.round(nifti1.get_fdata()).astype(np.uint8)
    data2 = np.round(nifti2.get_fdata()).astype(np.uint8)

    # Ensure the shapes match
    if data1.shape != data2.shape:
        raise ValueError("The two NIfTI files must have the same shape.")

    # Find unique classes in the images
    classes1 = np.unique(data1)
    classes2 = np.unique(data2)

    if classes1[0] == 0:
        classes1 = classes1[1:]
    if classes2[0] == 0:
        classes2 = classes2[1:]

    # Ensure both files have the same classes
    if not np.array_equal(classes1, classes2):
        raise ValueError("The two NIfTI files must have the same classes.")

    return nifti1, nifti2, classes1


def deform_segmentations(moving_segmentation: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """

    @param moving_segmentation:
    @param displacement:
    @return:
    """
    # TODO
