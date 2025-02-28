from pathlib import Path

from typing import Any, Tuple

import yaml
import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib

from registrationbaselines.core import utils
from registrationbaselines.core.types import floarArray4Dor5D, array2Dor3D
from registrationbaselines.displacement import utils_displacement


def load_displacement_niftyreg(path: Path) -> torch.Tensor:
    """
    Load a displacement field from a file and return it as a torch tensor in the shape H,W,D,3

    SHAPES:
    displacement_array.shape:	        (192, 138, 208, 1, 3)
    result (displacement_tensor.shape): (192, 138, 208, 3)

    @param path: The path to the displacement field file.

    @return: The displacement field.

    @raise ValueError: If the intent code of the displacement field is not NIFTI_INTENT_DISPVECT.
    @raise ValueError: If the displacement field has wrong dimensions.
    """

    displacement_nib = nib.load(path)
    displacement_array = displacement_nib.get_fdata()
    displacement_tensor = torch.from_numpy(displacement_array)

    displacement_tensor = displacement_tensor.squeeze(-2)

    shape = tuple(displacement_tensor.permute(2, 1, 0, 3).shape[:3])
    scaling_tensor = torch.tensor(shape).view(1, 1, 1, 3)
    displacement_tensor = (displacement_tensor / scaling_tensor) * 2  # nopep8

    displacement_tensor = displacement_tensor[..., [2, 1, 0]]

    return displacement_tensor.to(torch.float32)


def load_displacement(path: Path) -> torch.Tensor:
    """
    Load a displacement field from a file and return it as a torch tensor in the shape H,W,D,3

    @param path: The path to the displacement field file.

    @return: The displacement field.

    @raise ValueError: If the intent code of the displacement field is not NIFTI_INTENT_DISPVECT.
    @raise ValueError: If the displacement field has wrong dimensions.
    """

    utils.is_nifti(path)

    displacement_nib = nib.load(path)

    displacement_array = np.asarray(displacement_nib.dataobj)

    displacement_tensor = torch.from_numpy(
        displacement_array).squeeze().to(torch.float32)

    displacement_tensor = utils_displacement.displacement_to_unit_displacement(
        displacement_tensor)

    displacement_tensor = displacement_tensor[..., [2, 1, 0]]

    return displacement_tensor


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load a nifti image from a file and return it as a numpy array.

    The file should be in .nii or .nii.gz format.
    The voxel size should be isotropic.
    The direction should be identity.
    The image should be 3D.

    SHAPES:
    result (return_tensor.shape):         	(192, 138, 208)

    @param image_path: The path to the image file.

    @return: The image as a tensor.
    """

    utils.is_nifti(image_path)

    image_nib: nib.Nifti1Image = nib.load(image_path)

    utils.is_affine_identity(image_nib.affine)

    image_array: array2Dor3D = image_nib.get_fdata()

    dimension = image_array.ndim

    # check that it 3D
    if dimension != 3:
        raise ValueError(
            f"Dimension of {image_path} is not 3D: {dimension}"
        )

    return_tensor = torch.from_numpy(image_array).to(torch.float32)

    return return_tensor


def load_segmentation(image_path: Path) -> torch.Tensor:
    """
    Load a nifti segmentation from a file and return it as a numpy array.

    The file should be in .nii or .nii.gz format.
    The voxel size should be isotropic.
    The direction should be identity.
    The segmentation should be 3D.

    SHAPES:
    result (return_tensor.shape):         	(192, 138, 208)

    @param image_path: The path to the image file.

    @return: The image as a tensor.
    """

    utils.is_nifti(image_path)

    segmentation_nib: nib.Nifti1Image = nib.load(image_path)

    utils.is_affine_identity(segmentation_nib.affine)

    segmentation_array: array2Dor3D = segmentation_nib.get_fdata()

    dimension = segmentation_array.ndim

    # check that it 3D
    if dimension != 3:
        raise ValueError(
            f"Dimension of {image_path} is not 3D: {dimension}"
        )

    return_tensor = torch.from_numpy(segmentation_array).to(torch.uint8)

    return return_tensor


def load_keypoints(keypoints_path: Path) -> torch.Tensor:
    """
    Load keypoints from a csv file to a torch tensor of shape [N,3].

    dtype: float32

    The keypoints have to be in x,y,z order and they will be converted to z,y,x.
    The switch happens because extracting a np array from an sitk image does that too.

    @param keypoints_path: Path to the .txt keypoints file
    @return: Keypoints as a torch tensor of shape [N,3].
    """

    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")

    keypoints = np.loadtxt(keypoints_path, delimiter=',', dtype=np.float32)
    keypoints = torch.from_numpy(keypoints)

    if keypoints.ndim != 2:  # this is enforced by numpy, but let's keep it here
        raise ValueError(f"Keypoints should be 2D: {keypoints.ndim}")
    if keypoints.shape[1] != 3:
        raise ValueError(
            f"Keypoints should have 3 columns: {keypoints.shape[1]}")

    # switch x and z
    # keypoints[:, [0, 2]] = keypoints[:, [2, 0]]

    return keypoints


def read_config(file_path: Path) -> dict[str, Any]:
    """
    Read the configuration file.

    @param file_path: The path to the configuration file.
    @rtype file_path: Path

    @return: The configuration.
    @rtype: dict[str, Any]
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_image_spacing(image_path: Path) -> Tuple[float, float, float]:
    """
    Get the spacing of an image.

    @param image_path: The path to the image file.
    @type image_path: Path

    @return: The spacing.
    @rtype: Tuple[float, float, float]
    """

    utils.is_nifti(image_path)

    image_sitk: sitk.Image = sitk.ReadImage(image_path)

    spacing = image_sitk.GetSpacing()

    if len(spacing) != 3:
        raise ValueError(f"Spacing is not 3D: {spacing}")

    return spacing[2], spacing[1], spacing[1]


def get_image_shape(image_path: Path) -> Tuple[int, int, int]:

    utils.is_nifti(image_path)

    image_sitk: sitk.Image = sitk.ReadImage(image_path)

    image_array: array2Dor3D = sitk.GetArrayFromImage(image_sitk)

    image_tensor = torch.from_numpy(image_array).squeeze()

    image_tensor = image_tensor.permute(2, 1, 0)

    return image_tensor.shape
