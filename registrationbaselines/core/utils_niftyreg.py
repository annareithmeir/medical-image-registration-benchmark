from pathlib import Path
import os

from registrationbaselines.core import utils_commandline


def convert_control_point_grid_to_displacement_field(control_grid_path: Path,
                                                     fixed_path: Path) -> Path:
    """
    Helper function for NiftyReg.
    """

    assert control_grid_path.exists(
    ), f"File {control_grid_path} does not exist."
    assert fixed_path.exists(), f"File {fixed_path} does not exist."

    assert control_grid_path.suffix == '.nii' or control_grid_path.suffixes == ['.nii', '.gz'], \
        f"File {control_grid_path} is not a nifti file."
    assert fixed_path.suffix == '.nii' or fixed_path.suffixes == ['.nii', '.gz'], \
        f"File {fixed_path} is not a nifti file."

    # create command
    if control_grid_path.suffixes == ['.nii', '.gz']:
        path_displacement = Path(
            control_grid_path.as_posix().replace("_temp.nii", ".nii"))
    else:
        path_displacement = Path(
            control_grid_path.as_posix().replace("_temp.nii", ".nii"))

    path_reg_transform = Path(
        "registrationbaselines/libraries/NiftyReg/reg_transform_ubuntu").absolute()
    command_line_list = [path_reg_transform.as_posix(),
                         "-ref", fixed_path.as_posix(),
                         "-disp", control_grid_path.as_posix(),
                         path_displacement]

    utils_commandline.run_command_in_terminal(command_line_list,
                                              check=path_displacement.exists,
                                              print_command_list=True)

    # remove the temporary control point grid
    os.remove(control_grid_path)

    return path_displacement


def apply_transformation(path_fixed: Path, path_moving: Path, path_transfromation: Path) -> Path:
    """
    Apply the transformation to the moving image.~~~
    Works for both affine and non-linear transformations.
    """

    assert path_fixed.exists(), f"File {path_fixed} does not exist."
    assert path_moving.exists(), f"File {path_moving} does not exist."
    assert path_transfromation.exists(
    ), f"File {path_transfromation} does not exist."

    if path_transfromation.suffix == ".txt":
        method = "AffineNiftyReg"
    else:
        method = "BSplineNiftyReg"

    path_output, _ = utils_commandline.create_result_paths(
        path_fixed.parent, path_fixed.stem, path_moving.stem, method, ".nii", ".nii")

    path_reg_resample = Path(
        "registrationbaselines/libraries/NiftyReg/reg_resample_ubuntu").absolute()
    command = [path_reg_resample.as_posix(),
               "-ref", path_fixed.as_posix(),
               "-flo", path_moving.as_posix(),
               "-trans", path_transfromation.as_posix(),
               "-res", path_output.as_posix()]

    utils_commandline.run_command_in_terminal(command,
                                              check=path_output.exists,
                                              print_command_list=True)

    return path_output
