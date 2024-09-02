from pathlib import Path

from typing import List

from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg, utils_nifti
from registrationbaselines.data_loading import data_loaders


class BSplineNiftyReg(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True) -> None:

        super().__init__("BSplineNiftyReg",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation)

        self.path_reg_f3d = self.base_dir / Path(
            "registrationbaselines/libraries/NiftyReg/reg_f3d_ubuntu")

        # command to call NiftyReg
        self.command: List[str] = []

    def _register(self,
                  fixed_image_path: Path,
                  moving_image_path: Path) -> None:
        """
            Test
        """

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path
        self.working_dir_path = self.path_fixed.parent

        # check that both images exist
        assert self.path_fixed.exists(
        ), f"File {self.path_fixed} does not exist."
        assert self.path_moving.exists(
        ), f"File {self.path_moving} does not exist."

        self.__create_registration_command_list()
        utils_commandline.run_command_in_terminal(self.command,
                                                  self.__outputs_exist,
                                                  print_command_list=False)

        self.path_result_deformation = \
            utils_niftyreg.convert_transformation_to_displacement_field(
                self.result_control_grid_path, self.path_fixed)

        # assign intent code to the displacement field
        # utils_nifti.set_intent_code(
        #     self.path_result_deformation, "NIFTI_INTENT_DISPVECT")

    def __create_registration_command_list(self) -> None:
        """
        Create the command line list for the registration.
        """

        self.path_result_deformed, \
            self.result_control_grid_path = self._create_result_paths(self.path_fixed.stem,
                                                                      self.path_moving.stem,
                                                                      ".nii.gz",
                                                                      ".nii.gz")

        # control point grid is only temporary, we want to remove it later
        self.result_control_grid_path = Path(
            self.result_control_grid_path.as_posix().replace(".nii", "_temp.nii"))

        self.command = [self.path_reg_f3d.as_posix(),
                        '-ref', self.path_fixed.as_posix(),
                        '-flo', self.path_moving.as_posix(),
                        '-res', self.path_result_deformed.as_posix(),
                        '-cpp', self.result_control_grid_path.as_posix()]

        self.command = utils_commandline.add_configuration_to_command(self.command,
                                                                      self.run_configuration,
                                                                      only_value=True)

    def __outputs_exist(self) -> bool:
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.path_result_deformed.exists() and self.path_result_deformation.exists():
            return True

        return False
