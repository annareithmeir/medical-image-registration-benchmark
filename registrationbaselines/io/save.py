from pathlib import Path

from typing import Tuple

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import nibabel as nib

from registrationbaselines.displacement import utils_displacement
from registrationbaselines.core import utils_nifti


def save_displacement_niftyreg(displacement: torch.Tensor,
                               image_path: Path) -> None:

    displacement = displacement[..., [2, 1, 0]]

    shape = tuple(displacement.permute(2, 1, 0, 3).shape[:3])
    scaling_tensor = torch.tensor(shape).view(1, 1, 1, 3)
    displacement = (displacement / 2) * scaling_tensor  # nopep8

    displacement = displacement.unsqueeze(-2)

    displacement_nib = nib.Nifti1Image(
        displacement.detach().cpu().numpy(), np.eye(4))

    nib.save(displacement_nib, image_path)

    utils_nifti.set_intent_code(image_path, 'NIFTI_INTENT_DISPVECT')


def save_displacement(displacement: torch.Tensor,
                      image_path: Path,
                      spacing: Tuple[float, ...]) -> None:
    """
    Save a displacement field as a nifti image.

    The voxel size will be isotropic.
    The direction will be identity.

    SHAPES:
    entry (displacement_tensor.shape):         		(192, 138, 208, 3)
    numpy conversion (displacement_array.shape):	(3, 1, 208, 138, 192)
    result (displacement_sitk.GetSize()):	        (192, 138, 208, 1, 3)

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

    displacement_array = displacement.detach().cpu().numpy()

    sitk_displacement = sitk.GetImageFromArray(displacement_array)
    sitk_displacement.SetSpacing(spacing)

    sitk.WriteImage(sitk_displacement, image_path)

    utils_nifti.set_intent_code(image_path, 'NIFTI_INTENT_DISPVECT')


def save_image(image: torch.Tensor, image_path: Path) -> None:

    _save(image, image_path)


def save_segmentation(segmentation: torch.Tensor, image_path: Path) -> None:

    _save(segmentation, image_path)


def _save(image: torch.Tensor, image_path: Path) -> None:
    """
    Save a numpy array as a nifti image.

    The image can be 3D.
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

    volume_nib = nib.Nifti1Image(image.detach().cpu().numpy(), np.eye(4))

    nib.save(volume_nib, image_path)

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
