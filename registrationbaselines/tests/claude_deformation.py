import torch
import nibabel as nib
import sys
from pathlib import Path

from typing import Tuple

import numpy as np
import torch
import nibabel as nib


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.evaluation.plot_objects import plot_tensor_slices_difference  # nopep8
from registrationbaselines.warping import utils_displacement # nopep8

def load_nifti(file_path):
    """Load a NIfTI file and return its data as a numpy array."""
    nifti = nib.load(file_path)
    return nifti.get_fdata(), nifti.affine


def nifti_to_tensor(nifti_data):
    """Convert numpy array to PyTorch tensor and add batch and channel dimensions."""
    return torch.from_numpy(nifti_data).float().squeeze().unsqueeze(0)

def create_grid(shape):
    """Create a normalized grid for the given shape."""
    grid = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in shape], indexing='ij')
    return torch.stack(grid, dim=-1).unsqueeze(0)

def apply_affine_to_grid(grid, affine):
    """Apply affine transformation to the grid."""
    affine_tensor = torch.tensor(affine, dtype=torch.float32)
    grid_homogeneous = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    transformed_grid = torch.matmul(grid_homogeneous, affine_tensor.T)
    return transformed_grid[..., :3] / transformed_grid[..., 3:4]

def main(displacement_field_path, moving_image_path):
    # Load displacement field and moving image
    disp_field, disp_affine = load_nifti(displacement_field_path)
    moving_img, moving_affine = load_nifti(moving_image_path)

    # Convert to PyTorch tensors
    disp_field_tensor = nifti_to_tensor(disp_field)
    moving_img_tensor = nifti_to_tensor(moving_img).unsqueeze(0)

        # Normalize the displacement field
    disp_field_tensor = disp_field_tensor / torch.tensor(moving_img.shape).unsqueeze(0).unsqueeze(0).unsqueeze(0) * 2


    # Create normalized grid
    # grid = create_grid(moving_img.shape)
    grid = utils_displacement.compute_grid(moving_img_tensor.squeeze().shape,
                                           torch.float32)
                                           
    disp_field_tensor = disp_field_tensor[..., [2, 1, 0]]
    # Add displacement to the grid
    deformed_grid = grid + disp_field_tensor

    # Ensure the deformed grid is in the range [-1, 1]
    deformed_grid = torch.clamp(deformed_grid, -1, 1)

    # Perform deformation using grid_sample
    deformed_img = torch.nn.functional.grid_sample(
        moving_img_tensor, 
        deformed_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    ).squeeze()


    deformed_niftyreg_reg = torch.from_numpy(nib.load("registrationbaselines/tests/images/lung_deformed.nii.gz").get_fdata())

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
