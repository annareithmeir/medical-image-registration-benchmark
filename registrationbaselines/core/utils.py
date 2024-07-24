from pathlib import Path

from typing import Any, Union, Tuple

from scipy.ndimage import map_coordinates
import yaml
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from registrationbaselines.core import utils_metrics

from registrationbaselines.core.types import floatArray2D, floatArray3Dor4D, floarArray4Dor5D, array2Dor3D


def is_nifti_and_exists(path: Path) -> None:
    """
    Function to check if a file is a nifti and exists.

    @param path: The path to the file.

    @return: True if the file is a nifti and exists, False otherwise.

    @raise ValueError: If the file is not a nifti.
    @raise FileNotFoundError: If the file does not exist
    """

    # check that file is .nii or .nii.gz
    suffixes = path.suffixes
    if not suffixes == ['.nii'] and not suffixes == ['.nii', '.gz']:
        raise ValueError(
            "The file should be in .nii or .nii.gz format, but it is {suffixes}.")

    # check that file exists
    if not path.exists():
        raise FileNotFoundError(
            f"File {path} does not exist.")


def is_isotropic(image: sitk.Image) -> None:
    """
    Checks if an image has isotropic voxel size.

    @param image: The image.

    @raise ValueError: If the voxel size is not isotropic.
    """

    spacing = np.array(image.GetSpacing(), np.float64)

    if not np.allclose(spacing, spacing[0]):
        raise ValueError(
            f"Voxel size is not isotropic: {tuple(spacing)}"
        )


def is_direction_identity(image: sitk.Image) -> None:

    dimension = int(image.GetDimension())
    direction = np.abs(np.array(image.GetDirection(), np.float64))

    identity: np.ndarray[Any, np.dtype[np.float64]]
    match dimension:
        case 2:
            identity = np.eye(2)
        case 3:
            identity = np.eye(3)
        case 4:
            identity = np.eye(4)
        case _:
            identity = np.eye(5)

    if not np.allclose(direction, identity.flatten()):
        raise ValueError(
            f"Direction is not identity: {tuple(direction)}"
        )


def load_displacement(path: Path) -> torch.Tensor:
    """
    Load a displacement field from a file and return it as a numpy array.

    @param path: The path to the displacement field file.

    @return: The displacement field.

    @raise ValueError: If the intent code of the displacement field is not NIFTI_INTENT_DISPVECT.
    @raise ValueError: If the displacement field has wrong dimensions.
    """

    is_nifti_and_exists(path)

    displacement_sitk: sitk.Image = sitk.ReadImage(path)
    displacement_array: floarArray4Dor5D = sitk.GetArrayFromImage(
        displacement_sitk)

    # check intent code
    # 1006 = NIFTI_INTENT_DISPVECT
    if displacement_sitk.GetMetaData("intent_code") != "1006":
        raise ValueError(
            "The intent code of the displacement field should be NIFTI_INTENT_DISPVECT.")

    # is_isotropic(displacement_sitk)

    # check dimensions
    shape = displacement_array.shape

    # check that it is 4D or 5D
    if len(shape) not in [4, 5]:
        raise ValueError(
            f"Dimension is not 4D, 5D: {len(shape)}"
        )

    separating_dimension_correct = shape[1] == 1  # dim 1 is dummy
    # dim 0 is vector dimension, which has to correspond to spatial dimensions
    vector_dimension_correct = shape[0] == len(shape) - 2

    if not separating_dimension_correct or not vector_dimension_correct:
        raise ValueError(
            "The displacement field should have spatial dimensions as the last dimensions \
                and a vector dimension as the first dimension and separated by a dummy dimension.")

    displacement_tensor = torch.Tensor(displacement_array)

    # remove separating dummy dimension
    displacement_tensor = displacement_tensor.squeeze()

    # move the vector dimension to the last dimension
    new_order = list(range(1, displacement_tensor.dim())) + [0]
    displacement_tensor = displacement_tensor.permute(new_order)

    displacement_tensor = displacement_tensor.permute(2, 1, 0, 3)

    return displacement_tensor


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load a nifti image from a file and return it as a numpy array.

    The file should be in .nii or .nii.gz format.
    The voxel size should be isotropic.
    The direction should be identity.
    The image should be 2D, 3D or 4D.
    The image should be float or integer.

    @param image_path: The path to the image file.
    @type image_path: Path

    @return: The image.
    @rtype: floatArray2Dor3Dor4D
    """

    is_nifti_and_exists(image_path)

    image_sitk: sitk.Image = sitk.ReadImage(image_path)
    image_array: array2Dor3D = sitk.GetArrayFromImage(image_sitk)

    dimension = image_array.ndim

    # check that it is 2D, 3D or 4D
    if dimension not in [2, 3]:
        raise ValueError(
            f"Dimension of {image_path} is not 2D, 3D: {dimension}"
        )

    # check that spacing is isotropic
    # is_isotropic(image_sitk)

    # check that direction is identity
    is_direction_identity(image_sitk)

    return_tensor = torch.Tensor(image_array).squeeze()

    return_tensor = return_tensor.permute(2, 1, 0)

    return return_tensor


def save_image(image: torch.Tensor, image_path: Path, spacing: Tuple[int]) -> None:
    """
    Save a numpy array as a nifti image.

    The voxel size will be isotropic.
    The direction will be identity.
    The image can be 2D, 3D or 4D.
    """

    # check that file is .nii or .nii.gz
    if not image_path.suffix == '.nii' and not image_path.suffixes == ['.nii', '.gz']:
        raise ValueError(
            "The path should be in .nii or .nii.gz format.")

    # check that it is 2D, 3D or 4D
    if image.ndim not in [2, 3, 4]:
        raise ValueError(
            f"Dimension of image is not 2D, 3D or 4D: {image.ndim}"
        )

    sitk_image = sitk.GetImageFromArray(image.detach().cpu().numpy())
    sitk_image.SetSpacing(spacing)

    sitk.WriteImage(sitk_image, image_path)

    # check that file was written
    if not image_path.exists():
        raise FileNotFoundError(f"File {image_path} was not written.")


def get_affine_from_image(image: sitk.Image) -> floatArray2D:
    """
    Get the affine matrix from a SimpleITK image.

    @param image: The SimpleITK image.
    @type image: sitk.Image

    @return: The affine matrix.
    @rtype: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]
    """

    if image.GetDimension() != 3:
        raise ValueError("The image should be 3D.")

    direction = np.array(image.GetDirection(), np.float64).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), np.float64)
    origin = np.array(image.GetOrigin(), np.float64)

    # Construct the affine matrix
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction * spacing[:, None]
    affine[:3, 3] = origin

    return affine


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


def save_array_to_nii_gz_image(array: np.ndarray, filename: Path, affine: np.ndarray = None) -> None:
    """
    Saves a 3D numpy array to a .nii.gz file
    @param array: array
    @param filename: filename for daving
    @param affine: affine matrix of shaoe (4,4) 
    @return: 
    """
    assert array.ndim == 3
    image = sitk.GetImageFromArray(array)

    if affine is not None:
        # SimpleITK uses the direction cosine matrix, origin, and spacing to set the affine
        direction = affine[:3, :3].flatten()
        origin = affine[:3, 3]
        spacing = np.linalg.norm(affine[:3, :3], axis=0)

        image.SetDirection(direction)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

    sitk.WriteImage(image, str(filename))


def save_array_to_nii_gz_displacement_field(array: np.ndarray, filename: Path, affine: np.ndarray = None) -> None:
    """
    Saves a displacement field in form of np array to a .nii.gz file
    @param array: np array of shape (H,W,D,3)
    @param filename: filename for saving
    @param affine: affine matrix of shape (4,4)
    @return: 
    """
    assert array.ndim == 4
    assert array.shape[-1] == 3
    image = sitk.GetImageFromArray(array, isVector=True)

    if affine is not None:
        # SimpleITK uses the direction cosine matrix, origin, and spacing to set the affine
        direction = affine[:3, :3].flatten()
        origin = affine[:3, 3]
        spacing = np.linalg.norm(affine[:3, :3], axis=0)

        image.SetDirection(direction)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

    sitk.WriteImage(image, str(filename))


def displacement_to_unit_displacement(displacement: torch.Tensor) -> torch.Tensor:
    """
    Convert a displacement field to a unit displacement field.
    """

    for dim in range(displacement.shape[-1]):
        displacement[..., dim] = 2.0 * displacement[..., dim] / \
            float(displacement.shape[-dim - 2] - 1)

    return displacement


def deform_image(image: torch.Tensor,
                 displacement: torch.Tensor,
                 mode: str) -> torch.Tensor:
    """
    Apply a deformation to an image using the provided deformation.
    @param image_fixed:
    @param image_moving:
    @param displacement_field:
    @return:
    """

    # squeeze both image and displacement to ensure we only have spatial dimensions
    image = image.squeeze()
    displacement = displacement.squeeze()

    # convert to unit displacement if range is not [-1,1]
    if displacement.min() < -1 or displacement.max() > 1:
        displacement = displacement_to_unit_displacement(displacement)

    if image.ndim != displacement.ndim - 1:
        raise ValueError(
            "The displacement field should have one more dimension than the image.")

    grid = utils_metrics.compute_grid(
        image.shape, dtype=image.dtype, device=image.device)

    # unsqueeze image and displacement to conform to grid_sample requirements
    image = image.unsqueeze(0).unsqueeze(0)
    displacement = displacement.unsqueeze(0)

    # warp image
    warped_image = F.grid_sample(
        image, displacement + grid, mode=mode).squeeze()

    if warped_image.ndim != image.squeeze().ndim:
        raise ValueError(
            "The warped image should have the same number of dimensions as the original image. \
                Something wen wrong with deforming")

    return warped_image.squeeze()


def explore_memory():
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2)} MB")

    # Print the amount of reserved memory
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Reserved memory: {reserved_memory / (1024 ** 2)} MB")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory - allocated_memory

    print(f"Available memory: {available_memory / (1024 ** 2)} MB")
    print(f"Available memory: {available_memory / total_memory * 100} %")


def rgb_to_grayscale(rgb_image):
    r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
    grayscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale_image


def normalize_tensor_to_0_1(tensor: torch.tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def deform_keypoints(moving_keypoints: floatArray2D, displacement: floatArray3Dor4D) -> floatArray2D:
    """
    This works intyuitively, that is if at displacemente[10,10] you have a positive value, eg. 8,
    then the landmark at moving_keypoints[10,10] will be moved (or PUSHED, that's why intuitive) 8 units
    in the direction of the displacement. On the other hand, F.grid_sample works non-intuitively, that is
    it pulls - so 

    Map the moving keypoints to the fixed keypoints using the displacement field
    """

    if moving_keypoints.shape[-1] == 3:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, :, 0], moving_keypoints.transpose())
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, :, 1], moving_keypoints.transpose())
        mov_lms_disp_z = map_coordinates(
            displacement[:, :, :, 2], moving_keypoints.transpose())
        mov_lms_disp = np.array(
            (mov_lms_disp_x, mov_lms_disp_y, mov_lms_disp_z)).transpose()
    elif moving_keypoints.shape[-1] == 2:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, 0], moving_keypoints.transpose())
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, 1], moving_keypoints.transpose())
        mov_lms_disp = np.array((mov_lms_disp_x, mov_lms_disp_y)).transpose()
    else:
        raise ValueError(
            "The landmark shape is not supported. It should be either 2 or 3.")

    deformed_keypoints = moving_keypoints + mov_lms_disp

    assert isinstance(deformed_keypoints, np.ndarray)

    return deformed_keypoints
