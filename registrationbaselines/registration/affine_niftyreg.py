from pathlib import Path

from typing import List

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline


class AffineNiftyReg(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:

        self.method = "AffineNiftyReg"
        self.path_reg_aladin = Path("registrationbaselines/libraries/NiftyReg/reg_aladin_ubuntu").absolute()
        
        self.configuration = self.read_config(configuration_path)

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
        utils_commandline.run_command_in_terminal(self.command,
                                                  self._outputs_exist,
                                                  print_command_list=True)

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

        self.result_transformed_image_path, self.result_transformation_path = utils_commandline.create_result_paths(self.working_dir_path,
                                                                                                  self.fixed_path.stem,
                                                                                                  self.moving_path.stem,
                                                                                                  self.method,
                                                                                                  ".nii",
                                                                                                  ".txt")

        self.command = [self.path_reg_aladin.as_posix(),
                        '-ref', self.fixed_path.as_posix(),
                        '-flo', self.moving_path.as_posix(),
                        '-res', self.result_transformed_image_path.as_posix(),
                        '-aff', self.result_transformation_path.as_posix()]

        self.command = utils_commandline.add_configuration_to_command(self.command, self.configuration)

    def _outputs_exist(self):
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.result_transformed_image_path.exists() and self.result_transformation_path.exists():
            return True

        return False
    