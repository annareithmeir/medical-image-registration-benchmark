from pathlib import Path

from typing import List, Optional, Tuple

import torch
import SimpleITK as sitk

from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg, utils_nifti
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.io import save, load
from registrationbaselines.displacement import utils_displacement


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

        self.path_result_control_grid: Path

    def _register(self,
                  fixed_image: torch.Tensor,
                  moving_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            BUGFIX 0: The displacement field had to be adapted to our convention
                      Specifically:
                        - a different way of normalizing
                        - the spacing of the datastet has to be 1,1,1 (done in dataset preprocessing)
        """

        self.working_dir_path = Path(__file__).parent
        self.path_fixed = self.working_dir_path / "fixed.nii.gz"
        self.path_moving = self.working_dir_path / "moving.nii.gz"

        self.log(f"\t\tCreating temp files for NiftyReg")
        save.save_image(fixed_image, self.path_fixed)
        save.save_image(moving_image, self.path_moving)

        self.log(f"\t\tRunning NiftyReg in the terminal")
        self._create_registration_command_list()
        utils_commandline.run_command_in_terminal(self.command,
                                                  self._outputs_exist,
                                                  print_command_list=False)

        self.log(f"\t\tTransforming output to a displacement field")
        self.path_result_deformation = \
            utils_niftyreg.convert_transformation_to_displacement_field(
                self.path_result_control_grid, self.path_fixed)

        self.log(f"\t\tSetting the intent code of the displacement field")
        utils_nifti.set_intent_code(
            self.path_result_deformation, 'NIFTI_INTENT_DISPVECT')

        self.log(f"\t\tLoading the deformed image and the displacement field")
        deformed_image = load.load_image(self.path_result_deformed)
        displacement = load.load_displacement(
            self.path_result_deformation)

        self.log(f"\t\tRemoving temporary files")
        # remove temporary files
        self.path_fixed.unlink()
        self.path_moving.unlink()
        self.path_result_deformed.unlink()
        self.path_result_deformation.unlink()
        self.path_result_control_grid.unlink()

        return deformed_image, displacement

    def _create_registration_command_list(self) -> None:
        """
        Create the command line list for the registration.
        """

        self.path_result_deformed, \
            self.path_result_control_grid = self._create_result_paths(self.path_fixed.stem,
                                                                      self.path_moving.stem,
                                                                      ".nii.gz",
                                                                      ".nii.gz")

        # control point grid is only temporary, we want to remove it later
        self.path_result_control_grid = Path(
            self.path_result_control_grid.as_posix().replace(".nii", "_temp.nii"))

        self.command = [self.path_reg_f3d.as_posix(),
                        '-ref', self.path_fixed.as_posix(),
                        '-flo', self.path_moving.as_posix(),
                        '-res', self.path_result_deformed.as_posix(),
                        '-cpp', self.path_result_control_grid.as_posix()]

        self.command = utils_commandline.add_configuration_to_command(self.command,
                                                                      self.run_configuration,
                                                                      only_value=True)

    def _outputs_exist(self) -> bool:
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.path_result_deformed.exists():
            return True

        return False
