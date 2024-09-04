from pathlib import Path

from typing import List

from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg
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
            BUGFIX 0: The displacement field had to be adapted to out convention
                      Specifically:
                        - a different way of normalizing
                        - the spacing of the dataste has to be 1,1,1 (done in dataset preprocessing)
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

        # BUGFIX 0
        utils_niftyreg.convert_niftyreg_displacement_to_baseline_convention(
            self.path_result_deformation)

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

    # def _convert_niftyreg_displacement(self, path_deformation: Path) -> None:

    #     displacement_sitk = sitk.ReadImage(path_deformation)

    #     displacement_array = sitk.GetArrayFromImage(displacement_sitk)

    #     displacement_tensor = torch.from_numpy(displacement_array)

    #     displacement_tensor = utils_displacement.reverse_axis(
    #         displacement_tensor)

    #     displacement_tensor[:, :, :, 1] *= -1
    #     displacement_tensor[:, :, :, 2] *= -1

    #     # # should be unit displacement
    #     displacement_tensor = utils_displacement.displacement_to_unit_displacement(
    #         displacement_tensor)

    #     save.save_displacement(displacement_tensor,
    #                            path_deformation,
    #                            (1, 1, 1, 1))
