import shutil
from pathlib import Path

from typing import Tuple

import ants
import torch
import SimpleITK as sitk

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.displacement import utils_displacement
from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.io import save
from registrationbaselines.core import utils_nifti


class SyNANTs(RegistrationInterface):
    """
    SyN registration using ANTs.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True) -> None:
        """
        Initialize the registration model.

        NOTE: the displacement field won't work in slicer correctly if the correct itent code is set - the original should be left.
        """

        super().__init__("SyNANTs",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation)

    def _register(self,
                  fixed_image: torch.Tensor,
                  moving_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper around ants to register.

        BUGFIX 0: ants doesn't permute and flip on loading, so we have permute images on loading and permute and flip the displacement field on saving.
                  We also have to save the forward transform, as this was also used to register.
        """

        self.working_dir_path = Path(__file__).parent
        self.path_fixed = self.working_dir_path / "fixed.nii.gz"
        self.path_moving = self.working_dir_path / "moving.nii.gz"

        save.save_image(fixed_image, self.path_fixed)
        save.save_image(moving_image, self.path_moving)

        self.path_result_deformed, self.path_result_deformation = self._create_result_paths(self.path_fixed.stem,
                                                                                            self.path_moving.stem,
                                                                                            ".nii.gz",
                                                                                            ".nii.gz")

        # load boath images with ants
        fixed_image = ants.image_read(self.path_fixed.as_posix())
        moving_image = ants.image_read(self.path_moving.as_posix())

        # BUGFIX 0
        # premute x and z
        fixed_image = ants.from_numpy(fixed_image.numpy().transpose(2, 1, 0))
        moving_image = ants.from_numpy(moving_image.numpy().transpose(2, 1, 0))

        grad_step = self.run_configuration["grad_step"]
        flow_sigma = self.run_configuration["flow_sigma"]
        total_sigma = self.run_configuration["total_sigma"]

        # Perform registration
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            grad_step=grad_step,
            flow_sigma=flow_sigma,
            total_sigma=total_sigma,
            type_of_transform='SyNOnly',
            initial_transform="Identity",
            write_composite_transform=False  # nopep8 this outputs one .h5 transform, otherwise we have a .nii.gz and .mat
        )

        deformed = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                         transformlist=registration['fwdtransforms'])
        deformed_image = torch.from_numpy(deformed.numpy()).permute(2, 1, 0)

        displacement_sitk = sitk.ReadImage(registration['fwdtransforms'][0])
        displacement = torch.from_numpy(
            sitk.GetArrayFromImage(displacement_sitk))

        displacement = utils_displacement.displacement_to_unit_displacement(
            displacement)

        return deformed_image, displacement
