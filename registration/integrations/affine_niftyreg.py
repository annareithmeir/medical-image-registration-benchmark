from pathlib import Path
import subprocess
import os

from typing import List

from ..core.registration_interface import RegistrationInterface
from ..core.configurations import AffineNiftyRegConfiguration
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
    