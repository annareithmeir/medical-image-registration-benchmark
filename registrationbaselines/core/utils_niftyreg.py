import os
from pathlib import Path
import shutil

from typing import Optional

import SimpleITK as sitk
import torch

from registrationbaselines.core import utils_commandline, utils_nifti
from registrationbaselines.warping import utils_displacement


def convert_transformation_to_displacement_field(transformation_path: Path,
                                                 fixed_path: Path) -> Path:
    """
    Helper function for NiftyReg.
    """

    assert transformation_path.exists(
    ), f"File {transformation_path} does not exist."
    assert fixed_path.exists(), f"File {fixed_path} does not exist."

    assert fixed_path.suffix == '.nii' or fixed_path.suffixes == ['.nii', '.gz'], \
        f"File {fixed_path} is not a nifti file."

    # create command
    if transformation_path.suffixes == ['.txt']:
        path_displacement = Path(
            transformation_path.as_posix().replace("_temp.txt", ".nii.gz"))
    else:
        path_displacement = Path(
            transformation_path.as_posix().replace("_temp.nii", ".nii"))

    base_dir = Path(__file__).parent.parent.parent.absolute()
    path_reg_transform = Path(
        base_dir / "registrationbaselines/libraries/NiftyReg/reg_transform_ubuntu")
    command_line_list = [path_reg_transform.as_posix(),
                         "-ref", fixed_path.as_posix(),
                         "-disp", transformation_path.as_posix(),
                         path_displacement]

    utils_commandline.run_command_in_terminal(command_line_list,
                                              check=path_displacement.exists,
                                              print_command_list=False)

    # remove the temporary control point grid
    os.remove(transformation_path)

    return path_displacement


def apply_transformation(path_fixed: Path,
                         path_moving: Path,
                         path_transfromation: Path,
                         path_transformed: Path) -> Path:
    """
    Apply the transformation to the moving image.~~~
    Works for both affine and non-linear transformations.
    """

    assert path_fixed.exists(), f"File {path_fixed} does not exist."
    assert path_moving.exists(), f"File {path_moving} does not exist."
    assert path_transfromation.exists(
    ), f"File {path_transfromation} does not exist."

    base_dir = Path(__file__).parent.parent.parent.absolute()
    path_reg_resample = Path(
        base_dir / "registrationbaselines/libraries/NiftyReg/reg_resample_ubuntu")
    command = [path_reg_resample.as_posix(),
               "-ref", path_fixed.as_posix(),
               "-flo", path_moving.as_posix(),
               "-trans", path_transfromation.as_posix(),
               "-res", path_transformed.as_posix()]

    utils_commandline.run_command_in_terminal(command,
                                              check=path_transformed.exists,
                                              print_command_list=False)

    return path_transformed


def convert_niftyreg_displacement_to_baseline_convention(path_deformation: Path,
                                                         path_deformation_new: Optional[Path] = None) -> None:
    """
    Convert the displacement field from NiftyReg convention to the baseline convention.
        - normalises
        - converts to non-unit displacement field
        - permutes the vector dimension to the front
        - adds dummy dimension
        - saves
        - sets the intent code.

    If path_deformation_new is not provided, the original file will be overwritten.

    This function was introduced as a bugfix so that we can use the displacement field in our convention.
    TODO: this still doesn't work if the original images have non unit spacing

    @param path_deformation: Path to the displacement field in NiftyReg convention.
    @param path_deformation_new: Optional Path to save the new displacement field.

    @return: None
    """
    import numpy as np

    if not path_deformation_new:
        path_deformation_new = path_deformation

    # shutil.copy(path_deformation, path_deformation_new)
    utils_nifti.set_intent_code(path_deformation_new, 'NIFTI_INTENT_DISPVECT')
    """
    
    # get the displacement field as a torch tensor
    displacement_sitk = sitk.ReadImage(path_deformation_new)
    displacement_array = sitk.GetArrayFromImage(displacement_sitk)
    displacement_tensor = torch.from_numpy(displacement_array).to(torch.float64)

    displacement_tensor = displacement_tensor.squeeze(1)
    displacement_tensor = displacement_tensor.permute(1, 2, 3, 0)

    # Normalize the displacement field
    shape = tuple(displacement_tensor.permute(2, 1, 0, 3).shape[:3])
    scaling_tensor = torch.tensor(shape).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    displacement_tensor = (displacement_tensor / scaling_tensor)  * 2  # nopep8

    # convert it to a non-unit displacement field - because upon loading the displacement field
    # it will be converted back to a unit displacement field
    displacement_tensor = utils_displacement.unit_displacement_to_displacement(
        displacement_tensor)

    # move the vector dimension to the front to match sitk convention
    displacement_tensor = displacement_tensor.permute(3, 0, 1, 2)

    # add dummy dimension so the intent code can be set correctly
    # without it it will do some permutations to the tensor
    displacement_tensor = displacement_tensor.unsqueeze(1)
    
    # Save the new displacement field
    displacement_sitk_new = sitk.GetImageFromArray(displacement_tensor.numpy())

    sitk.WriteImage(displacement_sitk_new, path_deformation_new)
    """


def convert_niftyreg_displacement_to_baseline_convention_OLD(path_deformation: Path,
                                                             path_deformation_new: Optional[Path] = None) -> None:
    """
    Convert the displacement field from NiftyReg convention to the baseline convention.
        - normalises
        - converts to non-unit displacement field
        - permutes the vector dimension to the front
        - adds dummy dimension
        - saves
        - sets the intent code.

    If path_deformation_new is not provided, the original file will be overwritten.

    This function was introduced as a bugfix so that we can use the displacement field in our convention.
    TODO: this still doesn't work if the original images have non unit spacing

    @param path_deformation: Path to the displacement field in NiftyReg convention.
    @param path_deformation_new: Optional Path to save the new displacement field.

    @return: None
    """

    if not path_deformation_new:
        path_deformation_new = path_deformation

    # get the displacement field as a torch tensor
    displacement_sitk = sitk.ReadImage(path_deformation)
    displacement_array = sitk.GetArrayFromImage(displacement_sitk)
    displacement_tensor = torch.from_numpy(displacement_array)

    # Normalize the displacement field
    shape = tuple(displacement_tensor.permute(2, 1, 0, 3).shape[:3])
    displacement_tensor = displacement_tensor / \
        torch.tensor(shape).unsqueeze(
            0).unsqueeze(0).unsqueeze(0) * 2

    # convert it to a non-unit displacement field - because upon loading the displacement field
    # it will be converted back to a unit displacement field
    displacement_tensor = utils_displacement.unit_displacement_to_displacement(
        displacement_tensor)

    # move the vector dimension to the front to match sitk convention
    displacement_tensor = displacement_tensor.permute(3, 0, 1, 2)

    # add dummy dimension so the intent code can be set correctly
    # without it it will do some permutations to the tensor
    displacement_tensor = displacement_tensor.unsqueeze(1)

    # Save the new displacement field
    displacement_sitk_new = sitk.GetImageFromArray(displacement_tensor.numpy())

    sitk.WriteImage(displacement_sitk_new, path_deformation_new)

    utils_nifti.set_intent_code(path_deformation_new, 'NIFTI_INTENT_DISPVECT')
