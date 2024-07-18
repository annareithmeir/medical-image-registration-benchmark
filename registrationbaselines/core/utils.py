from pathlib import Path

from typing import Any

from scipy.ndimage import map_coordinates
import yaml
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from registrationbaselines.core import utils_metrics

from registrationbaselines.core.types import floatArray2D, floatArray3Dor4D, floatArray2Dor3Dor4D


def load_image(image_path: Path) -> floatArray2Dor3Dor4D:
    """
    Load a nifti image from a file and return it as a numpy array.

    The file should be in .nii or .nii.gz format.
    The voxel size should be isotropic.
    The direction should be identity.
    The image should be 2D, 3D or 4D.

    @param image_path: The path to the image file.
    @type image_path: Path

    @return: The image.
    @rtype: floatArray2Dor3Dor4D
    """

    # check that file is .nii or .nii.gz
    if not image_path.suffix == '.nii' and not image_path.suffix == '.nii.gz':
        raise TypeError(
            "The image file should be in .nii or .nii.gz format.")

    image: sitk.Image = sitk.ReadImage(image_path)

    # check that it is 2D, 3D or 4D
    dimension = int(image.GetDimension())
    if dimension not in [2, 3, 4]:
        raise ValueError(
            f"Dimension of {image_path} is not 2D, 3D or 4D: {dimension}"
        )

    # check that spacing is isotropic
    spacing = np.array(image.GetSpacing(), np.float64)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError(
            f"Voxel size of {image_path} is not isotropic: {tuple(spacing)}"
        )

    # check that direction is identity
    direction = np.array(image.GetDirection(), np.float64)

    identity: np.ndarray[Any, np.dtype[np.float64]]
    match dimension:
        case 2:
            identity = np.eye(2)
        case 3:
            identity = np.eye(3)
        case _:
            identity = np.eye(4)

    if not np.allclose(direction, identity.flatten()):
        raise ValueError(
            f"Direction of {image_path} is not identity: {tuple(direction)}"
        )

    image_array = sitk.GetArrayFromImage(image)

    return image_array


def get_affine_from_image(image: sitk.Image) -> floatArray2D:
    """
    Get the affine matrix from a SimpleITK image.

    @param image: The SimpleITK image.
    @type image: sitk.Image

    @return: The affine matrix.
    @rtype: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]
    """

    direction = np.array(image.GetDirection(), np.float64).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), np.float64)
    origin = np.array(image.GetOrigin(), np.float64)

    # Construct the affine matrix
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction * spacing[:, None]
    affine[:3, 3] = origin

    return affine


def read_config(file_path: Path) -> dict:
    """
    Read the configuration file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_image_from_nii_gz(image_path: Path) -> np.ndarray:
    """
    Loads a .nii.gz file to a numpy array
    @param image_path: path of image
    @return: numpy array
    """
    image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(image)


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


def deform_image(image: torch.Tensor,
                 displacement: torch.Tensor, mode) -> torch.Tensor:
    """
    Apply a deformation to an image using the provided deformation.
    @param image_fixed:
    @param image_moving:
    @param displacement_field:
    @return:
    """

    image_size = image.shape[-2:]

    grid = utils_metrics.compute_grid(
        image_size, dtype=image.dtype, device=image.device)

    if displacement.shape[0] != 1:
        displacement = displacement.unsqueeze(0)

    if image.shape[0] != 1:
        image = image.unsqueeze(0)
    if image.shape[1] != 1:
        image = image.unsqueeze(1)

    # warp image
    warped_image = F.grid_sample(image, - displacement + grid, mode=mode)

    return warped_image


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


def deform_landmarks(moving_landmarks: floatArray2D, displacement: floatArray3Dor4D) -> floatArray2D:
    """
    This works intyuitively, that is if at displacemente[10,10] you have a positive value, eg. 8,
    then the landmark at moving_landmarks[10,10] will be moved (or PUSHED, that's why intuitive) 8 units
    in the direction of the displacement. On the other hand, F.grid_sample works non-intuitively, that is
    it pulls - so 

    Map the moving landmarks to the fixed landmarks using the displacement field
    """

    if moving_landmarks.shape[-1] == 3:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, :, 0], moving_landmarks.transpose())
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, :, 1], moving_landmarks.transpose())
        mov_lms_disp_z = map_coordinates(
            displacement[:, :, :, 2], moving_landmarks.transpose())
        mov_lms_disp = np.array(
            (mov_lms_disp_x, mov_lms_disp_y, mov_lms_disp_z)).transpose()
    elif moving_landmarks.shape[-1] == 2:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, 0], moving_landmarks.transpose())
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, 1], moving_landmarks.transpose())
        mov_lms_disp = np.array((mov_lms_disp_x, mov_lms_disp_y)).transpose()
    else:
        raise ValueError(
            "The landmark shape is not supported. It should be either 2 or 3.")

    deformed_landmarks = moving_landmarks + mov_lms_disp

    assert isinstance(deformed_landmarks, np.ndarray)

    return deformed_landmarks
