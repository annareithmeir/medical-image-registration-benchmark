from pathlib import Path
import os

from registrationbaselines.core import utils_commandline


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
