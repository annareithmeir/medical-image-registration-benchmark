from pathlib import Path
import subprocess
import os

from typing import List

import nibabel as nib

from ._interface_registration import RegistrationInterface
from ..core.configurations import BSplineNiftyRegConfiguration
from ..core.enums import TransformationType
from ..core.utilities import create_result_paths


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


F3D_PATH = Path(os.path.expanduser('~/bin/reg_f3d'))


class BSplineNiftyReg(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_registration: BSplineNiftyRegConfiguration):

        self.method = "BSplineNiftyReg"

        # registration configuration
        self.transfromation_type = configuration_registration.transformation_type

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_control_grid_path = Path()
        self.result_transformation_path = Path()
        self.working_dir_path = Path()

        # remaining arguments
        self.remaining_arguments = configuration_registration.remaining_arguments

        # command to call NiftyReg
        self.command: List[str] = []

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False) -> None:
        """
            Test
        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path
        self.working_dir_path = self.fixed_path.parent

        # check that both images exist
        assert self.fixed_path.exists(
        ), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(
        ), f"File {self.moving_path} does not exist."

        self._create_registration_command_list()

        self._print_command_line(self.command)

        try:
            p = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = p.communicate()

            if p.returncode != 0 or not self._outputs_exist():

                error_message = ''
                if not self._outputs_exist():
                    error_message += 'Outputs not written on the disk\n\n'
                error_message += str(output[1])

                raise FileNotFoundError(error_message)

        except OSError as e:
            print(e)
            print('Is blockmatching correctly installed?')
        
        self.result_transformation_path = self._convert_control_point_grid_to_displacement_field()

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        return self.result_transformation_path

    def _create_registration_command_list(self):
        """
        Create the command line list for the registration.
        """

        self.result_transformed_image_path, self.result_control_grid_path = create_result_paths(self.working_dir_path,
                                                                                                  self.fixed_path.stem,
                                                                                                  self.moving_path.stem,
                                                                                                  self.method,
                                                                                                  ".nii",
                                                                                                  ".nii")

        self.command = [F3D_PATH.as_posix()]
        self.command += ['-ref', self.fixed_path.as_posix()]
        self.command += ['-flo', self.moving_path.as_posix()]
        self.command += ['-res', self.result_transformed_image_path.as_posix()]

        if self.transfromation_type == TransformationType.B_SPLINE:
            # control point grid is only temporary, we want to remove it later
            self.result_control_grid_path = Path(self.result_control_grid_path.as_posix().replace(".nii", "_temp.nii"))

            self.command += ['-cpp', self.result_control_grid_path]
        else:
            raise ValueError(
                f"Transformation type {self.transfromation_type} not supported.")
        
        if self.remaining_arguments is not None:
            self.command += self.remaining_arguments

    def _print_command_line(self, cmd_list):
        print('\n\n')

        full_cmd = ""

        for a in cmd_list:
            full_cmd += str(a) + " "

        print(full_cmd)

    def _outputs_exist(self):
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.result_transformed_image_path.exists() and self.result_transformation_path.exists():
            return True

        return False
    
    def _convert_control_point_grid_to_displacement_field(self) -> Path:

        assert self.result_control_grid_path.exists(), f"File {self.result_control_grid_path} does not exist."
        assert self.fixed_path.exists(), f"File {self.fixed_path} does not exist."

        assert self.result_control_grid_path.suffix == '.nii' or self.result_control_grid_path.suffixes == ['.nii', '.gz'], \
            f"File {self.result_control_grid_path} is not a nifti file."
        assert self.fixed_path.suffix == '.nii' or self.fixed_path.suffixes == ['.nii', '.gz'], \
            f"File {self.fixed_path} is not a nifti file."

        # create command
        if self.result_control_grid_path.suffixes == ['.nii', '.gz']:
            path_displacement = Path(
                self.result_control_grid_path.as_posix().replace("_temp.nii", ".nii"))
        else:
            path_displacement = Path(
                self.result_control_grid_path.as_posix().replace("_temp.nii", ".nii"))

        command_line_list = ["reg_transform", "-ref", self.fixed_path.as_posix(), "-disp",
                            self.result_control_grid_path.as_posix(), path_displacement]

        self._print_command_line(command_line_list)

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

        self._set_intent_code(path_displacement, 'NIFTI_INTENT_DISPVECT')

        # remove the temporary control point grid
        os.remove(self.result_control_grid_path)

        return path_displacement


    def _set_intent_code(self, path: Path, intent_code: str) -> None:
        """
        Set the intent code of a nifti file.

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
