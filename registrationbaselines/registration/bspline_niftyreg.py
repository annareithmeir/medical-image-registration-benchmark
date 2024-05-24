from pathlib import Path

from typing import List

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg


class BSplineNiftyReg(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:

        self.method = "BSplineNiftyReg"
        self.base_dir = Path(__file__).parent.parent.absolute().parent
        self.path_reg_f3d = self.base_dir / Path(
            "registrationbaselines/libraries/NiftyReg/reg_f3d_ubuntu")

        self.configuration = self.read_config(configuration_path)

        self._create_result_directories()

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_control_grid_path = Path()
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

        self.__create_registration_command_list()
        utils_commandline.run_command_in_terminal(self.command,
                                                  self.__outputs_exist,
                                                  print_command_list=False)

        self.result_transformation_path = utils_niftyreg.convert_control_point_grid_to_displacement_field(
            self.result_control_grid_path, self.fixed_path)

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        return self.result_transformation_path

    def _save_results(self, deformed, deformation):
        """
        Nothing happens here because saving is done thorugh the command line.
        """

    def __create_registration_command_list(self):
        """
        Create the command line list for the registration.
        """

        self.result_transformed_image_path, self.result_control_grid_path = self._create_result_paths(self.fixed_path.stem,
                                                                                                      self.moving_path.stem,
                                                                                                      ".nii",
                                                                                                      ".nii")

        # control point grid is only temporary, we want to remove it later
        self.result_control_grid_path = Path(
            self.result_control_grid_path.as_posix().replace(".nii", "_temp.nii"))

        self.command = [self.path_reg_f3d.as_posix(),
                        '-ref', self.fixed_path.as_posix(),
                        '-flo', self.moving_path.as_posix(),
                        '-res', self.result_transformed_image_path.as_posix(),
                        '-cpp', self.result_control_grid_path.as_posix()]

        self.command = utils_commandline.add_configuration_to_command(
            self.command, self.configuration)

    def __outputs_exist(self):
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.result_transformed_image_path.exists() and self.result_transformation_path.exists():
            return True

        return False
