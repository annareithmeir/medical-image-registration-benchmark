from pathlib import Path

from typing import List, Optional, Tuple

import torch
import SimpleITK as sitk

from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg, utils_nifti
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.io import save, load
from registrationbaselines.displacement import utils_displacement


class uniGradICON(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True) -> None:

        super().__init__("uniGradICON",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation)
        
        self.modality = dataloader.modality

        # command to call uniGradICON
        self.command: List[str] = []

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

        save.save_image(fixed_image, self.path_fixed)
        save.save_image(moving_image, self.path_moving)

        self._create_registration_command_list()
        utils_commandline.run_command_in_terminal(self.command,
                                                  self._outputs_exist,
                                                  print_command_list=False)

        utils_nifti.set_intent_code(
            self.path_result_deformation, 'NIFTI_INTENT_DISPVECT')

        deformed_image = load.load_image(self.path_result_deformed)
        displacement = load.load_displacement(
            self.path_result_deformation)

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
            self.path_result_deformation = self._create_result_paths(self.path_fixed.stem,
                                                                      self.path_moving.stem,
                                                                      ".nii.gz",
                                                                      ".nii.gz")

        self.command = ["unigradicon-register",
                        '--fixed=', self.path_fixed.as_posix(),
                        '--fixed_modality', self.path_moving.as_posix(),
                        '--moving=', self.path_moving.as_posix(),
                        '--moving_modality', self.path_moving.as_posix(),
        
--fixed_modality=ct --moving=sftp:/data/LungCT_preprocessed_new/imagesTr/LungCT_0001_0001.nii.gz --moving_modality=ct --transform_out=trans.nii.gz --warped_moving_out=/u/home/koeglf/temp/temp_unigradicon/warped.nii.gz --io_iterations None
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
