from pathlib import Path
import subprocess
import os

from typing import List

from registrationbaselines.core.registration_interface import RegistrationInterface
from registrationbaselines.core.configurations import AffineNiftyRegConfiguration
from registrationbaselines.core.enums import TransformationType
from registrationbaselines.core.utilities import create_result_paths

ALADIN_PATH = Path(os.path.expanduser('~/bin/reg_aladin'))


class AffineNiftyReg(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_registration: AffineNiftyRegConfiguration):

        self.method = "AffineNiftyReg"

        # registration configuration
        self.transfromation_type = configuration_registration.transformation_type

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
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

        self._print_command_line()

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

        self.result_transformed_image_path, self.result_transformation_path = create_result_paths(self.working_dir_path,
                                                                                                  self.fixed_path.stem,
                                                                                                  self.moving_path.stem,
                                                                                                  self.method,
                                                                                                  ".nii",
                                                                                                  ".txt")

        self.command = [ALADIN_PATH.as_posix()]
        self.command += ['-ref', self.fixed_path.as_posix()]
        self.command += ['-flo', self.moving_path.as_posix()]
        self.command += ['-res', self.result_transformed_image_path.as_posix()]

        if self.transfromation_type == TransformationType.RIGID:
            self.command += ['-rigOnly']
        elif self.transfromation_type == TransformationType.AFFINE:
            self.command += ['-affDirect']
        else:
            raise ValueError(
                f"Transformation type {self.transfromation_type} not supported.")

        self.command += ['-aff', self.result_transformation_path.as_posix()]

        if self.remaining_arguments is not None:
            self.command += self.remaining_arguments

    def _print_command_line(self):
        print('\n\n')

        full_cmd = ""

        for a in self.command:
            full_cmd += a + " "

        print(full_cmd)

    def _outputs_exist(self):
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.result_transformed_image_path.exists() and self.result_transformation_path.exists():
            return True

        return False
    