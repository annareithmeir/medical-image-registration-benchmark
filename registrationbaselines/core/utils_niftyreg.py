import subprocess
from pathlib import Path
import os

import nibabel as nib

import utils


# intent codes for nifti files - at the moment we only need NIFTI_INTENT_DISPVECT
INTENT_CODES = ['NIFTI_INTENT_CORREL', 'NIFTI_INTENT_TTEST', 'NIFTI_INTENT_FTEST',
                'NIFTI_INTENT_ZSCORE', 'NIFTI_INTENT_CHISQ', 'NIFTI_INTENT_BETA',
                'NIFTI_INTENT_BINOM', 'NIFTI_INTENT_GAMMA', 'NIFTI_INTENT_POISSON',
                'NIFTI_INTENT_NORMAL', 'NIFTI_INTENT_FTEST_NONC', 'NIFTI_INTENT_CHISQ_NONC',
                'NIFTI_INTENT_LOGISTIC', 'NIFTI_INTENT_LAPLACE', 'NIFTI_INTENT_UNIFORM',
                'NIFTI_INTENT_TTEST_NONC', 'NIFTI_INTENT_WEIBULL', 'NIFTI_INTENT_CHI',
                'NIFTI_INTENT_INVGAUSS', 'NIFTI_INTENT_EXTVAL', 'NIFTI_INTENT_PVAL',
                'NIFTI_INTENT_LOGPVAL', 'NIFTI_INTENT_LOG10PVAL', 'NIFTI_FIRST_STATCODE',
                'NIFTI_LAST_STATCODE', 'NIFTI_INTENT_ESTIMATE', 'NIFTI_INTENT_LABEL',
                'NIFTI_INTENT_NEURONAME', 'NIFTI_INTENT_GENMATRIX', 'NIFTI_INTENT_SYMMATRIX',
                'NIFTI_INTENT_DISPVECT', 'NIFTI_INTENT_VECTOR', 'NIFTI_INTENT_POINTSET',
                'NIFTI_INTENT_TRIANGLE', 'NIFTI_INTENT_QUATERNION', 'NIFTI_INTENT_DIMLESS']

REG_TRANSFORM_PATH = Path('/usr/local/bin/reg_transform')


def set_intent_code(path: Path, intent_code: str) -> None:
    """
    Set the intent code of a nifti file. This is necessary when using a displacement field from NiftyReg - then we need to add NIFTI_INTENT_DISPVECT

    Parameters
    ----------
    path : Path
        Path to the nifti file.
    intent_code : str
        The intent code to set.
    """

    assert path.exists(), f"File {path} does not exist."
    assert path.suffix == '.nii' or path.suffix == '.nii.gz', \
        f"File {path} is not a nifti file."
    assert intent_code in INTENT_CODES, \
        f"Intent code {intent_code} is not valid."

    image = nib.load(path)

    # we have to construct a new image with the new intent code, otherwise we cannot overwrite
    header = image.header
    data = image.get_fdata()

    # set the code
    header.set_intent(nib.nifti1.intent_codes[intent_code])

    # create new image
    new_image = nib.Nifti1Image(data, image.affine, header)

    # save the new image
    try:
        nib.save(new_image, path.as_posix())
    except Exception as e:
        print(f"Error saving the file: {e}")


def convert_affine_to_displacement_field(path_fixed: Path,
                                                  path_transformation: Path) -> Path:
    """
    Helper function for NiftyReg.
    """

    assert path_transformation.suffixes == ['.txt'], f"Transformation file {path_transformation} is not a txt file."

    path_output = Path(path_transformation.as_posix().replace(".txt", ".nii"))

    command = [REG_TRANSFORM_PATH.as_posix()]
    command.extend(["-ref", path_fixed.as_posix()])
    command.extend(["-disp", path_transformation.as_posix()])
    command.extend([path_output.as_posix()])

    utils.print_command(command)

    try:
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = p.communicate()

        if not path_output.exists():

            error_message = 'Outputs not written on the disk\n\n'
            error_message += str(output[1])

            raise FileNotFoundError(error_message)

    except OSError as e:
        print(e)
        print('Is reg_transform correctly installed?')

    set_intent_code(path_output, 'NIFTI_INTENT_DISPVECT')

    return path_output


def convert_control_point_grid_to_displacement_field(control_grid_path: Path,
                                                     fixed_path: Path) -> Path:
    """
    Helper function for NiftyReg.
    """

    assert control_grid_path.exists(), f"File {control_grid_path} does not exist."
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

    command_line_list = ["reg_transform", "-ref", fixed_path.as_posix(), "-disp",
                        control_grid_path.as_posix(), path_displacement]

    utils.print_command(command_line_list)

    try:
        p = subprocess.Popen(
            command_line_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = p.communicate()

        if not path_displacement.exists():

            error_message = "Output volume not written on the disk\n\n"
            error_message += output[1]
            raise FileNotFoundError(error_message)
    except OSError as e:
        print(e)
        print('Is reg_transform correctly installed?')

    set_intent_code(path_displacement, 'NIFTI_INTENT_DISPVECT')

    # remove the temporary control point grid
    os.remove(control_grid_path)

    return path_displacement