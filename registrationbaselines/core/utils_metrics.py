from pathlib import Path

from typing import Tuple, Any

import numpy as np
import SimpleITK as sitk
import torch


def compute_grid(image_size, dtype=torch.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")


def extract_class(segmentation: np.ndarray[Any, Any], class_value: int):
    """
    Extract a specific class from the segmentation.

    @param segmentation: The segmentation data.
    @type  segmentation: C{np.ndarray}
    @param class_value: The class value to extract.
    @type  class_value: C{int}

    @return: A binary mask where the class value is set to 1 and others to 0.
    @rtype:  C{np.ndarray}
    """

    assert isinstance(
        segmentation, np.ndarray), "Segmentation must be a numpy array."
    assert isinstance(class_value, int), "Class value must be an integer."

    return (segmentation == class_value).astype(np.uint8)


def save_class_nifti(original_img: sitk.Image,
                     class_mask: np.ndarray,
                     output_path: str) -> None:
    """
    Save a binary mask as a NIfTI file.

    Args:
    original_img (sitk.Image): The original NIfTI image.
    class_mask (np.ndarray): The mask for the class.
    output_path (str): The output file path.

    Returns:
    None
    """

    # change to binary mask
    class_mask[class_mask > 0] = 1

    class_img = sitk.GetImageFromArray(class_mask)
    class_img.SetOrigin(original_img.GetOrigin())
    class_img.SetSpacing(original_img.GetSpacing())
    class_img.SetDirection(original_img.GetDirection())

    sitk.WriteImage(class_img, output_path)


def get_segmentation_classes(nifti_path: Path) -> np.ndarray:
    """
    Load a NIfTI file and return the unique classes in the image.

    params:
    nifti_path (Path): Path to the NIfTI file.
    """

    # Load the NIfTI file
    nifti = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path.as_posix()))

    # Find unique classes in the image
    classes = np.unique(nifti)

    if classes[0] == 0.0:
        classes = classes[1:]

    return classes


def get_maks_and_classes(nifti1_path: Path, nifti2_path: Path) -> Tuple[sitk.Image, sitk.Image, np.ndarray]:
    """
    Load two NIfTI files and return them and unique classes - class 0 is removed.

    params:
    nifti1_path (Path): Path to the first NIfTI file.
    nifti2_path (Path): Path to the second NIfTI file.
    remove_zero_class (bool): Whether to remove the zero class from the images (background class).
    """

    # Load the NIfTI files
    nifti1 = sitk.ReadImage(nifti1_path.as_posix())
    nifti2 = sitk.ReadImage(nifti2_path.as_posix())

    # round each value to nearest integer
    data1 = np.round(sitk.GetArrayFromImage(nifti1)).astype(np.uint8)
    data2 = np.round(sitk.GetArrayFromImage(nifti2)).astype(np.uint8)

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
