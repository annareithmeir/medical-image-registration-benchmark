from pathlib import Path
import yaml
import numpy as np
import SimpleITK as sitk
import torch

def read_config(file_path: Path):
    """
    Read the configuration file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_image_from_nii_gz(image_path:Path) -> np.ndarray:
    """
    Loads a .nii.gz file to a numpy array
    @param image_path: path of image
    @return: numpy array
    """
    image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(image)


def save_array_to_nii_gz_image(array: np.ndarray, filename: Path, affine: np.ndarray=None) -> None:
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


def save_array_to_nii_gz_displacement_field(array: np.ndarray, filename: Path, affine: np.ndarray=None) -> None:
    """
    Saves a displacement field in form of np array to a .nii.gz file
    @param array: np array of shape (H,W,D,3)
    @param filename: filename for saving
    @param affine: affine matrix of shape (4,4)
    @return: 
    """
    assert array.ndim == 4
    assert array.shape[-1]==3
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


def apply_displacement_field(image_fixed: np.ndarray,
                             image_moving: np.ndarray,
                             displacement_field: np.ndarray,
                             sitk_interpolator: int) -> np.ndarray:
    """
    Apply a deformation to an image using the provided deformation.
    @param image_fixed:
    @param image_moving:
    @param displacement_field:
    @param sitk_interpolator:
    @return:
    """

    image_fixed = sitk.GetImageFromArray(image_fixed)
    image_moving = sitk.GetImageFromArray(image_moving)
    displacement_field = sitk.GetImageFromArray(displacement_field, isVector=True)

    # Create the transform using the displacement field
    displacement_field_transform = sitk.DisplacementFieldTransform(
        displacement_field)

    # Apply the transform to the input image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_fixed)
    resampler.SetInterpolator(sitk_interpolator)
    resampler.SetTransform(displacement_field_transform)

    deformed_image = resampler.Execute(image_moving)

    return sitk.GetArrayFromImage(deformed_image)


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

