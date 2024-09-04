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


def load_nifti(file_path):
    """Load a NIfTI file and return its data as a numpy array."""
    nifti = nib.load(file_path)
    return nifti.get_fdata()


def nifti_to_tensor(nifti_data):
    """Convert numpy array to PyTorch tensor and add batch and channel dimensions."""
    return torch.from_numpy(nifti_data).float().squeeze().unsqueeze(0)


def convert_niftyreg_displacement(path_deformation_old: Path,
                                  path_deformation_new: Path,
                                  shape: Tuple[int, int, int]) -> None:
    displacement_sitk = sitk.ReadImage(path_deformation_old)

    displacement_array = sitk.GetArrayFromImage(displacement_sitk)

    displacement_tensor = torch.from_numpy(displacement_array)

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


def main(displacement_field_path, moving_image_path):

    displacement_field_path_new = Path(
        "registrationbaselines/tests/images/lung_deformation_converted.nii.gz")

    moving_img = load_nifti(moving_image_path)

    convert_niftyreg_displacement(Path(displacement_field_path),
                                  displacement_field_path_new,
                                  moving_img.shape)

    # Load displacement field and moving image
    disp_field = load_nifti(displacement_field_path_new)

    # Convert to PyTorch tensors
    disp_field_tensor = nifti_to_tensor(disp_field)
    moving_img_tensor = nifti_to_tensor(moving_img).unsqueeze(0)

    grid = utils_displacement.compute_grid(moving_img_tensor.squeeze().shape,
                                           torch.float32)

    # Add displacement to the grid
    deformed_grid = grid + disp_field_tensor

    # Ensure the deformed grid is in the range [-1, 1]
    deformed_grid_clamped = deformed_grid  # torch.clamp(deformed_grid, -1, 1)

    # Perform deformation using grid_sample
    deformed_img = torch.nn.functional.grid_sample(
        moving_img_tensor,
        deformed_grid_clamped,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze()

    deformed_img_ours = deform_objects.deform_image(
        torch.from_numpy(moving_img).to(torch.float32),
        disp_field_tensor,
        grid,
        disp_field_tensor)

    deformed_niftyreg_reg = torch.from_numpy(nib.load(
        "registrationbaselines/tests/images/lung_deformed.nii.gz").get_fdata())

    plot_tensor_slices_difference(deformed_img,
                                  deformed_niftyreg_reg,
                                  "registrationbaselines/tests/images/diff.jpg")

    print("Deformation completed successfully.")
    print(f"Original shape: {moving_img.shape}")
    print(f"Deformed shape: {deformed_img.shape}")

    # Here you can add code to visualize or save the deformed image
    # For example, using matplotlib or saving as a new NIfTI file


if __name__ == "__main__":
    displacement_field_path = "registrationbaselines/tests/images/lung_deformation.nii.gz"
    moving_image_path = "registrationbaselines/tests/images/LungCT_0001_0001.nii.gz"
    main(displacement_field_path, moving_image_path)
