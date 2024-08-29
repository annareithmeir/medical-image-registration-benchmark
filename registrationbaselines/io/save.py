from pathlib import Path

from typing import Tuple

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch

from registrationbaselines.warping import utils_displacement


def save_displacement(displacement: torch.Tensor,
                      image_path: Path,
                      spacing: Tuple[float, ...]) -> None:
    """
    Save a displacement field as a nifti image.

    The voxel size will be isotropic.
    The direction will be identity.

    BUGFIX_0: we have to reverse the axis of the displacement (and in the spacing),
              to match the reversal in loading
    """
    from registrationbaselines.core import utils_nifti

    # check that file is .nii or .nii.gz
    if not image_path.suffix == '.nii' and not image_path.suffixes == ['.nii', '.gz']:
        raise ValueError(
            "The path should be in .nii or .nii.gz format.")

    # check dimensions
    shape = displacement.shape

    # check that it is 4D
    if len(shape) != 4:
        raise ValueError(f"Dimension of displacement is not 4D: {len(shape)}")
    if shape[-1] != 3:
        raise ValueError(
            "The displacement field should have the vector dimension as the last dimension.")

    if displacement.dtype != torch.float32:
        raise TypeError(
            f"Dsiplacement is not torch.float32: {displacement.dtype}"
        )

    # spacing has to match the image
    if len(spacing) != len(shape):
        raise ValueError(
            "The spacing does not match the image dimensions."
        )

    # BUGFIX_0
    displacement = utils_displacement.reverse_axis(displacement)

    # should be unit displacement
    displacement = utils_displacement.unit_displacement_to_displacement(
        displacement)

    # move vector dimension from back to front
    displacement = displacement.permute(3, 0, 1, 2)
    spacing = (spacing[-1],) + spacing[:-1]

    # insert separating dimension
    displacement = displacement.unsqueeze(1)

    # add dummy spacing
    spacing = (spacing[0], 1) + spacing[1:]

    # BUGFIX_0
    # reverse spacing
    spacing = (spacing[4], spacing[3], spacing[2], spacing[1], spacing[0])

    sitk_displacement = sitk.GetImageFromArray(
        displacement.detach().cpu().numpy())
    sitk_displacement.SetSpacing(spacing)

    sitk.WriteImage(sitk_displacement, image_path)

    utils_nifti.set_intent_code(image_path, 'NIFTI_INTENT_DISPVECT')


def save_image(image: torch.Tensor, image_path: Path, spacing: Tuple[float, ...]) -> None:
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

    # check that it is 3D
    if image.ndim != 3:
        raise ValueError(
            f"Dimension of image is not 3D: {image.ndim}"
        )

    # spacing has to match the image
    if len(spacing) != image.ndim:
        raise ValueError(
            "The spacing does not match the image dimensions."
        )

    image = image.permute(2, 1, 0)
    spacing = list(reversed(spacing))

    sitk_image = sitk.GetImageFromArray(image.detach().cpu().numpy())
    sitk_image.SetSpacing(spacing)

    sitk.WriteImage(sitk_image, image_path)

    # check that file was written
    if not image_path.exists():
        raise FileNotFoundError(f"File {image_path} was not written.")


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


def transform_csv(input_csv: Path, output_csv: Path) -> None:
    """
    Transform to a csv that can be read by Slicer
    """

    # Load the CSV file using np.genfromtxt
    coords = np.genfromtxt(input_csv, delimiter=',')

    # Prepare the data for the new format
    data = {
        'label': [f'F-{i+1}' for i in range(coords.shape[0])],
        'l': coords[:, 0],
        'p': coords[:, 1],
        's': coords[:, 2],
        'defined': [1] * coords.shape[0],
        'selected': [1] * coords.shape[0],
        'visible': [1] * coords.shape[0],
        'locked': [0] * coords.shape[0],
        'description': [''] * coords.shape[0]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to the output CSV file
    df.to_csv(output_csv, index=False)
