
from pathlib import Path

from typing import Any, Dict, Optional

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch
import torch.nn.functional as F

from registrationbaselines.warping import utils_displacement


def deform_image(image: torch.Tensor,
                 displacement: torch.Tensor) -> torch.Tensor:
    """
    Apply a deformation to an image using the provided deformation.
    If the image is of type uint8 ie a segmentation map,
    it automatically uses mode='nearest' and returns an image of type uint8
    @param image: if label map then dtype must be torch.uint8, else torch.float
    @param displacement:
    @return:
    """

    # squeeze both image and displacement to ensure we only have spatial dimensions
    image = image.squeeze()
    displacement = displacement.squeeze()

    # convert to unit displacement if range is not [-1,1]
    if not utils_displacement.is_unit_displacement(displacement):
        displacement = utils_displacement.displacement_to_unit_displacement(
            displacement)

    if image.ndim != displacement.ndim - 1:
        raise ValueError(
            "The displacement field should have one more dimension than the image.")

    if image.dtype == torch.uint8:
        mode = 'nearest'
        image = image.float()
    elif image.dtype == torch.float32:
        mode = 'bilinear'
    else:
        raise ValueError(
            "The image should be either uint8 or float32.")

    grid = utils_displacement.compute_grid(
        image.shape, dtype=image.dtype, device=image.device)

    # unsqueeze image and displacement to conform to grid_sample requirements
    image = image.unsqueeze(0).unsqueeze(0)
    displacement = displacement.unsqueeze(0)

    # warp image
    warped_image = F.grid_sample(
        image, displacement + grid, mode=mode, align_corners=True).squeeze()

    if warped_image.ndim != image.squeeze().ndim:
        raise ValueError(
            "The warped image should have the same number of dimensions as the original image. \
                Something wen wrong with deforming")

    if mode == 'nearest':
        warped_image = warped_image.to(dtype=torch.uint8)
    return warped_image.squeeze()


def deform_keypoints(moving_keypoints: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """
    Deforms keypoints according to the pull convention using grid_sample.

    @param moving_keypoints: Tensor of shape (N, 3) where N is the number of keypoints (18 in this case).
    @param displacement: Tensor of shape (201, 201, 201, 3) containing the displacement field.
    @return: Deformed keypoints as a Tensor of shape (N, 3).
    """

    if utils_displacement.is_unit_displacement(displacement):
        displacement = utils_displacement.unit_displacement_to_displacement(
            displacement)

    N = moving_keypoints.shape[0]

    moving_keypoints = moving_keypoints[:, [2, 1, 0]]

    # Normalize moving_keypoints to the range [-1, 1] for grid_sample
    grid = moving_keypoints.unsqueeze(0)  # Shape (1, N, 3)

    # Normalize the grid to [-1, 1] based on the displacement field size
    grid = (grid - torch.tensor([displacement.shape[0] / 2, displacement.shape[1] / 2, displacement.shape[2] / 2], device=grid.device)) \
        / torch.tensor([displacement.shape[0] / 2, displacement.shape[1] / 2, displacement.shape[2] / 2], device=grid.device)

    # Reshape the grid to the correct shape for grid_sample
    grid = grid.view(1, 1, 1, N, 3)  # Shape (1, 1, 1, N, 3)

    # Prepare displacement field for grid_sample
    displacement = displacement.permute(3, 0, 1, 2).unsqueeze(
        0)  # Shape (1, 3, 201, 201, 201)

    # Use grid_sample to sample the displacement field at the keypoints' locations
    sampled_displacement = F.grid_sample(
        displacement, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Reshape the sampled displacement to match the original keypoints shape
    sampled_displacement = sampled_displacement.squeeze().transpose(0, 1)  # Shape (N, 3)

    # Apply the displacement to the keypoints
    deformed_keypoints = moving_keypoints - sampled_displacement  # Pull convention

    deformed_keypoints = deformed_keypoints[:, [2, 1, 0]]

    return deformed_keypoints


def deform_keypointsOLD(moving_keypoints: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """
    Deforms keypoints according to the pull convention

    Map the moving keypoints to the fixed keypoints using the displacement field
    The displacement field should be pixel-based for this to work, so in case it is a unit-displacement field, it is first converted...
    @param moving_keypoints:
    @param displacement: of shape (...,3) and optimally non-unit displacement (will be converted otherwise)
    @return:
    """

    if utils_displacement.is_unit_displacement(displacement):
        displacement = utils_displacement.unit_displacement_to_displacement(
            displacement)

    # Transpose the keypoints to match the shape for map_coordinates (3, N)
    moving_keypoints_t = moving_keypoints.transpose(0, 1)

    if moving_keypoints.shape[-1] == 3:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, :, 0], moving_keypoints_t)
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, :, 1], moving_keypoints_t)
        mov_lms_disp_z = map_coordinates(
            displacement[:, :, :, 2], moving_keypoints_t)
        mov_lms_disp = torch.tensor(
            (mov_lms_disp_x, mov_lms_disp_y, mov_lms_disp_z)).transpose(0, 1)
    elif moving_keypoints.shape[-1] == 2:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, 0], moving_keypoints_t)
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, 1], moving_keypoints_t)
        mov_lms_disp = torch.tensor(
            (mov_lms_disp_x, mov_lms_disp_y)).transpose(0, 1)
    else:
        raise ValueError(
            "The landmark shape is not supported. It should be either 2 or 3.")

    """
######################################################################################################
    ###### TEMPORARY  ################
######################################################################################################
    # Step 2: Get the integer coordinates of landmarks
    moving_coords = moving_keypoints.long()

    # Ensure the coordinates are within the valid range
    moving_coords[:, 0] = torch.clamp(
        moving_coords[:, 0], 0, displacement.shape[1] - 1)
    moving_coords[:, 1] = torch.clamp(
        moving_coords[:, 1], 0, displacement.shape[2] - 1)
    moving_coords[:, 2] = torch.clamp(
        moving_coords[:, 2], 0, displacement.shape[3] - 1)

    displacement = unit_displacement_to_displacement(displacement)

    # Step 3: Extract the displacements for each landmark
    displacements = displacement[moving_coords[:, 0],
                                 moving_coords[:, 1],
                                 moving_coords[:, 2], :]

    # Step 4: Apply the displacements
    displaced_landmarks_zyx = moving_coords.float() - displacements

    return displaced_landmarks_zyx
    """

    deformed_keypoints = moving_keypoints - mov_lms_disp  # pull
    return deformed_keypoints


def deform_image_niftyreg_path(path_image: Path,
                               path_deformation: Path) -> Path:

    from registrationbaselines.core import utils_commandline

    path_warped_image = Path(
        path_image.as_posix().replace(".nii", "_warpedManually.nii"))

    command = ["/u/home/koeglf/Documents/code/registrationbaselines/registrationbaselines/libraries/NiftyReg/reg_resample_ubuntu",
               '-ref', path_image.as_posix(),
               '-flo', path_image.as_posix(),
               '-trans', path_deformation.as_posix(),
               '-res', path_warped_image.as_posix()]

    utils_commandline.run_command_in_terminal(command,
                                              path_warped_image.exists,
                                              print_command_list=False)

    return path_warped_image


def deform_image_niftyreg_nibabel(image: nib.Nifti1Image,
                                  deformation: nib.Nifti1Image) -> nib.Nifti1Image:
    path_image = Path("image.nii.gz")
    nib.save(image, path_image)

    path_deformation = Path("deformation.nii.gz")
    nib.save(deformation, path_deformation)

    path_warped_image = deform_image_niftyreg_path(path_image,
                                                   path_deformation)

    warped = nib.load(path_warped_image)

    path_image.unlink()
    path_deformation.unlink()

    return warped


def deform_image_niftyreg_sitk(image: sitk.Image,
                               deformation: sitk.Image,
                               image_path: Optional[Path] = None,
                               deformation_path: Optional[Path] = None) -> sitk.Image:

    if image_path:
        path_image = image_path
    else:
        path_image = Path("image.nii.gz")
        sitk.WriteImage(image, path_image)

    if deformation_path:
        path_deformation = deformation_path
    else:
        path_deformation = Path("deformation.nii.gz")
        sitk.WriteImage(deformation, path_deformation)

    path_warped_image = deform_image_niftyreg_path(path_image,
                                                   path_deformation)

    warped = sitk.ReadImage(path_warped_image)

    if not image_path:
        path_image.unlink()
    if not deformation_path:
        path_deformation.unlink()

    return warped


def deform_image_niftyreg_numpy(image: np.ndarray[Any, Any],
                                deformation: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:

    image_nib = nib.Nifti1Image(image.astype(np.float32), affine=None)
    deformation_nib = nib.Nifti1Image(deformation, affine=None)

    warped_nib = deform_image_niftyreg_nibabel(image_nib, deformation_nib)

    warped = warped_nib.get_fdata(dtype=np.float32)

    return warped


"""
def deform_image_niftyreg_numpy(image: np.ndarray[Any, Any],
                                deformation: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:

    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    sitk_deformation = sitk.GetImageFromArray(deformation)

    sitk_warped = deform_image_niftyreg_sitk(sitk_image, sitk_deformation)

    warped = sitk.GetArrayFromImage(sitk_warped).astype(np.float32)

    return warped
"""

"""
def deform_image_niftyreg_torch(image: torch.Tensor,
                                deformation: torch.Tensor) -> torch.Tensor:

    sitk_image = sitk.GetImageFromArray(
        image.detach().cpu().numpy().astype(np.float32))
    sitk_deformation = sitk.GetImageFromArray(
        deformation.detach().cpu().numpy())

    sitk_warped = deform_image_niftyreg_sitk(sitk_image, sitk_deformation)

    warped = torch.from_numpy(
        sitk.GetArrayFromImage(sitk_warped).astype(np.float32))

    return warped
"""


def deform_image_niftyreg_torch(image: torch.Tensor,
                                deformation: torch.Tensor) -> torch.Tensor:

    image_np = image.detach().cpu().numpy().astype(np.float32)
    deformation_np = deformation.detach().cpu().numpy()

    image_nib = nib.Nifti1Image(image_np, affine=None)
    deformation_nib = nib.Nifti1Image(deformation_np, affine=None)

    warped_nib = deform_image_niftyreg_nibabel(image_nib, deformation_nib)

    warped = torch.from_numpy(warped_nib.get_fdata(dtype=np.float32))

    return warped


def register_niftyreg(
    path_fixed: Path,
    path_moving: Path,
    path_result_deformed: Path,
    path_result_deformation: Path
) -> None:

    from registrationbaselines.core import utils_commandline, utils_niftyreg

    # check that both images exist
    assert path_fixed.exists(
    ), f"File {path_fixed} does not exist."
    assert path_moving.exists(
    ), f"File {path_moving} does not exist."

    path_reg_f3d = Path(
        "/u/home/koeglf/Documents/code/registrationbaselines/registrationbaselines/libraries/NiftyReg/reg_f3d_ubuntu")

    path_result_gird = path_moving.parent / "deformation_temp.nii.gz"

    command = [path_reg_f3d.as_posix(),
               '-ref', path_fixed.as_posix(),
               '-flo', path_moving.as_posix(),
               '-res', path_result_deformed.as_posix(),
               '-cpp', path_result_gird.as_posix()]

    utils_commandline.run_command_in_terminal(command,
                                              path_result_deformed.exists,
                                              print_command_list=False)

    path_result_deformation_temp = utils_niftyreg.convert_transformation_to_displacement_field(
        path_result_gird, path_fixed)

    path_result_deformation_temp.rename(path_result_deformation)
