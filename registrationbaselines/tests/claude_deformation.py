import torch
import nibabel as nib
import sys
from pathlib import Path

from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.evaluation.plot_objects import plot_tensor_slices_difference  # nopep8
from registrationbaselines.warping import utils_displacement, deform_objects  # nopep8
from registrationbaselines.io import load, save  # nopep8
from registrationbaselines.core import utils_nifti  # nopep8


def load_nifti_image(file_path) -> torch.Tensor:
    """Load a NIfTI file and return its data as a numpy array."""

    image_sitk = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image_sitk)

    return_tensor = torch.from_numpy(image_array).squeeze()

    return_tensor = return_tensor.permute(2, 1, 0)

    return return_tensor.to(torch.float32)


def load_nifti_displacement(file_path) -> torch.Tensor:
    """Load a NIfTI file and return its data as a numpy array."""
    nib_image = nib.load(file_path)
    nib_array = nib_image.get_fdata()
    nib_tensor = torch.from_numpy(nib_array).squeeze()

    return nib_tensor.to(torch.float32)


def load_nifti_displacement_sitk(file_path) -> torch.Tensor:

    displacement_sitk = sitk.ReadImage(file_path)

    displacement_array = sitk.GetArrayFromImage(displacement_sitk)

    displacement_tensor = torch.from_numpy(displacement_array).squeeze()

    displacement_tensor = displacement_tensor.permute(1, 2, 3, 0)

    # should be unit displacement
    displacement_tensor = utils_displacement.displacement_to_unit_displacement(
        displacement_tensor)

    displacement_tensor = utils_displacement.reverse_axis(displacement_tensor)

    return displacement_tensor.to(torch.float32)


def convert_niftyreg_displacement(path_deformation_old: Path,
                                  path_deformation_new: Path) -> None:
    displacement_sitk = sitk.ReadImage(path_deformation_old)

    displacement_array = sitk.GetArrayFromImage(displacement_sitk)

    displacement_tensor = torch.from_numpy(displacement_array)

    shape = tuple(displacement_tensor.permute(2, 1, 0, 3).shape[:3])

    # Normalize the displacement field
    displacement_tensor = displacement_tensor / \
        torch.tensor(shape).unsqueeze(
            0).unsqueeze(0).unsqueeze(0) * 2

    # Create normalized grid
    displacement_tensor = displacement_tensor[..., [2, 1, 0]]

    # Save the new displacement field
    displacement_sitk_new = sitk.GetImageFromArray(displacement_tensor.numpy())
    displacement_sitk_new.CopyInformation(displacement_sitk)
    sitk.WriteImage(displacement_sitk_new, path_deformation_new)


def convert_niftyreg_displacement_new(path_deformation_old: Path,
                                      path_deformation_new: Path) -> None:
    displacement_sitk = sitk.ReadImage(path_deformation_old)

    displacement_array = sitk.GetArrayFromImage(displacement_sitk)

    displacement_tensor = torch.from_numpy(displacement_array)

    shape = tuple(displacement_tensor.permute(2, 1, 0, 3).shape[:3])

    # Normalize the displacement field
    displacement_tensor = displacement_tensor / \
        torch.tensor(shape).unsqueeze(
            0).unsqueeze(0).unsqueeze(0) * 2

    # Create normalized grid
    # displacement_tensor = displacement_tensor[..., [2, 1, 0]]

    displacement_tensor = displacement_tensor.permute(3, 0, 1, 2)
    # displacement_tensor: torch.Tensor = utils_displacement.reverse_axis(displacement_tensor)
    displacement_tensor = displacement_tensor.unsqueeze(1)

    # Save the new displacement field
    displacement_sitk_new = sitk.GetImageFromArray(displacement_tensor.numpy())

    sitk.WriteImage(displacement_sitk_new, path_deformation_new)

    utils_nifti.set_intent_code(path_deformation_new, 'NIFTI_INTENT_DISPVECT')






def main(displacement_field_path, moving_image_path) -> None:

    displacement_field_path_convert = Path(
        "registrationbaselines/tests/images/lung_deformation_converted.nii.gz")
    displacement_field_path_convert_new = Path(
        "registrationbaselines/tests/images/lung_deformation_converted_new.nii.gz")

    moving_img_tensor = load_nifti_image(moving_image_path)

    convert_niftyreg_displacement(Path(displacement_field_path),
                                  displacement_field_path_convert)
    convert_niftyreg_displacement_new(Path(displacement_field_path),
                                  displacement_field_path_convert_new)

    # Load displacement field and moving image
    disp_field_tensor_old = load_nifti_displacement(displacement_field_path_convert) # nopep8
    disp_field_tensor = load.load_displacement(displacement_field_path_convert_new)  # nopep8
    diff = torch.sum(torch.abs(disp_field_tensor - disp_field_tensor_old))
    disp_field_tensor = disp_field_tensor.unsqueeze(0)

    grid = utils_displacement.compute_grid(moving_img_tensor.squeeze().shape,
                                           torch.float32)

    # Add displacement to the grid
    deformed_grid = grid + disp_field_tensor

    # Ensure the deformed grid is in the range [-1, 1]
    deformed_grid_clamped = deformed_grid  # torch.clamp(deformed_grid, -1, 1)

    moving_img_tensor = moving_img_tensor.unsqueeze(0).unsqueeze(0)

    # Perform deformation using grid_sample
    deformed_img = torch.nn.functional.grid_sample(
        moving_img_tensor,
        deformed_grid_clamped,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze()

    deformed_img_ours = deform_objects.deform_image(
        moving_img_tensor,
        disp_field_tensor,
        grid,
        disp_field_tensor)

    deformed_niftyreg_reg = load_nifti_image(
        "registrationbaselines/tests/images/lung_deformed.nii.gz").squeeze()

    plot_tensor_slices_difference(deformed_img,
                                  deformed_niftyreg_reg,
                                  "registrationbaselines/tests/images/diff_claude.jpg")

    plot_tensor_slices_difference(deformed_img_ours,
                                  deformed_niftyreg_reg,
                                  "registrationbaselines/tests/images/diff_ours.jpg")

    print("Deformation completed successfully.")


if __name__ == "__main__":
    displacement_field_path = "registrationbaselines/tests/images/lung_deformation.nii.gz"
    moving_image_path = "registrationbaselines/tests/images/LungCT_0001_0001.nii.gz"
    main(displacement_field_path, moving_image_path)
