from pathlib import Path

from typing import Any, Tuple

import yaml
import numpy as np
import SimpleITK as sitk
import torch

from registrationbaselines.core import utils
from registrationbaselines.core.types import floarArray4Dor5D, array2Dor3D
from registrationbaselines.warping import utils_displacement


def load_displacement(path: Path) -> torch.Tensor:
    """
    Load a displacement field from a file and return it as a torch tensor in the shape H,W,D,3

    SHAPES:
    entry (displacement_sitk.GetSize()):            (192, 138, 208, 1, 3)
    numpy conversion (displacement_array.shape):	(3, 1, 208, 138, 192)
    result (displacement_tensor.shape):         	(192, 138, 208, 3)

    @param path: The path to the displacement field file.

    @return: The displacement field.

    @raise ValueError: If the intent code of the displacement field is not NIFTI_INTENT_DISPVECT.
    @raise ValueError: If the displacement field has wrong dimensions.
    """
    """
    utils.is_nifti(path)

    displacement_sitk: sitk.Image = sitk.ReadImage(path)

    utils.is_isotropic(displacement_sitk)
    utils.is_direction_identity(displacement_sitk)
    utils.are_offdiagonal_direction_elements_zero(displacement_sitk)

    displacement_array: floarArray4Dor5D = sitk.GetArrayFromImage(
        displacement_sitk)

    # check intent code
    # 1006 = NIFTI_INTENT_DISPVECT
    if displacement_sitk.GetMetaData("intent_code") != "1006":
        raise ValueError(
            "The intent code of the displacement field should be NIFTI_INTENT_DISPVECT.")

    # check dimensions
    shape = displacement_array.shape

    # check that it is 5D
    if len(shape) != 5:
        raise ValueError(f"Dimension is not 5D: {len(shape)}")

    separating_dimension_correct = shape[1] == 1  # dim 1 is dummy
    # dim 0 is vector dimension, which has to correspond to spatial dimensions
    vector_dimension_correct = shape[0] == len(shape) - 2

    if not separating_dimension_correct or not vector_dimension_correct:
        raise ValueError(
            "The displacement field should have spatial dimensions as the last dimensions \
                and a vector dimension as the first dimension and separated by a dummy dimension.")

    displacement_tensor = torch.from_numpy(displacement_array)
    if displacement_tensor.dtype != torch.float32:
        raise TypeError(
            f"Dsiplacement is not torch.float32: {displacement_tensor.dtype}")
    """

    displacement_sitk = sitk.ReadImage(path)
    displacement_array = sitk.GetArrayFromImage(displacement_sitk)
    displacement_tensor = torch.from_numpy(displacement_array)

    displacement_tensor = displacement_tensor.squeeze(1)
    displacement_tensor = displacement_tensor.permute(1, 2, 3, 0)

    # Normalize the displacement field
    shape = tuple(displacement_tensor.permute(2, 1, 0, 3).shape[:3])
    scaling_tensor = torch.tensor(shape).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    displacement_tensor = (displacement_tensor / scaling_tensor) * 2  # nopep8

    # convert it to a non-unit displacement field - because upon loading the displacement field
    # it will be converted back to a unit displacement field
    displacement_tensor = utils_displacement.unit_displacement_to_displacement(
        displacement_tensor)

    # move the vector dimension to the front to match sitk convention
    displacement_tensor = displacement_tensor.permute(3, 0, 1, 2)

    # add dummy dimension so the intent code can be set correctly
    # without it it will do some permutations to the tensor
    displacement_tensor = displacement_tensor.unsqueeze(1)

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    # remove separating dummy dimension
    displacement_tensor = displacement_tensor.squeeze()

    # move the vector dimension to the last dimension
    new_order = list(range(1, displacement_tensor.dim())) + [0]
    displacement_tensor = displacement_tensor.permute(new_order)

    # should be unit displacement
    displacement_tensor = utils_displacement.displacement_to_unit_displacement(
        displacement_tensor)

    displacement_tensor = utils_displacement.reverse_axis(displacement_tensor)

    return displacement_tensor


def load_displacement_OLD(path: Path) -> torch.Tensor:
    """
    Load a displacement field from a file and return it as a torch tensor in the shape H,W,D,3

    @param path: The path to the displacement field file.

    @return: The displacement field.

    @raise ValueError: If the intent code of the displacement field is not NIFTI_INTENT_DISPVECT.
    @raise ValueError: If the displacement field has wrong dimensions.
    """

    utils.is_nifti(path)

    displacement_sitk: sitk.Image = sitk.ReadImage(path)

    utils.is_isotropic(displacement_sitk)
    utils.is_direction_identity(displacement_sitk)
    utils.are_offdiagonal_direction_elements_zero(displacement_sitk)

    displacement_array: floarArray4Dor5D = sitk.GetArrayFromImage(
        displacement_sitk)

    # check intent code
    # 1006 = NIFTI_INTENT_DISPVECT
    if displacement_sitk.GetMetaData("intent_code") != "1006":
        raise ValueError(
            "The intent code of the displacement field should be NIFTI_INTENT_DISPVECT.")

    # check dimensions
    shape = displacement_array.shape

    # check that it is 5D
    if len(shape) != 5:
        raise ValueError(f"Dimension is not 5D: {len(shape)}")

    separating_dimension_correct = shape[1] == 1  # dim 1 is dummy
    # dim 0 is vector dimension, which has to correspond to spatial dimensions
    vector_dimension_correct = shape[0] == len(shape) - 2

    if not separating_dimension_correct or not vector_dimension_correct:
        raise ValueError(
            "The displacement field should have spatial dimensions as the last dimensions \
                and a vector dimension as the first dimension and separated by a dummy dimension.")

    displacement_tensor = torch.from_numpy(displacement_array)
    if displacement_tensor.dtype != torch.float32:
        raise TypeError(
            f"Dsiplacement is not torch.float32: {displacement_tensor.dtype}")

    # remove separating dummy dimension
    displacement_tensor = displacement_tensor.squeeze()

    # move the vector dimension to the last dimension
    new_order = list(range(1, displacement_tensor.dim())) + [0]
    displacement_tensor = displacement_tensor.permute(new_order)

    # should be unit displacement
    displacement_tensor = utils_displacement.displacement_to_unit_displacement(
        displacement_tensor)

    displacement_tensor = utils_displacement.reverse_axis(displacement_tensor)

    return displacement_tensor


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load a nifti image from a file and return it as a numpy array.

    The file should be in .nii or .nii.gz format.
    The voxel size should be isotropic.
    The direction should be identity.
    The image should be 2D, 3D or 4D.
    The image should be float or integer.

    SHAPES:
    entry (image_sitk.GetSize()):           (192, 138, 208)
    numpy conversion (image_array.shape):	(208, 138, 192)
    result (return_tensor.shape):         	(192, 138, 208)

    @param image_path: The path to the image file.

    @return: The image as a tensor.
    """

    utils.is_nifti(image_path)

    image_sitk: sitk.Image = sitk.ReadImage(image_path)

    utils.is_isotropic(image_sitk)
    utils.is_direction_identity(image_sitk)
    utils.are_offdiagonal_direction_elements_zero(image_sitk)

    image_array: array2Dor3D = sitk.GetArrayFromImage(image_sitk)

    dimension = image_array.ndim

    # check that it 3D
    if dimension != 3:
        raise ValueError(
            f"Dimension of {image_path} is not 3D: {dimension}"
        )

    return_tensor = torch.from_numpy(image_array).squeeze()
    # check that image is float or int
    if not (return_tensor.dtype == torch.uint8 or return_tensor.dtype == torch.float32):
        raise TypeError(
            f"Image is not float or int: {return_tensor.dtype}"
        )

    return_tensor = return_tensor.permute(2, 1, 0)

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
