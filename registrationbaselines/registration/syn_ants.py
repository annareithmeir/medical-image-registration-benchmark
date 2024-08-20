import shutil

import ants
from pathlib import Path
import wandb
from typing import Dict, Any
import torch

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_nifti, utils


class SyNANTs(RegistrationInterface):
    """
    SyN registration using ANTs.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset) -> None:
        """
        Initialize the registration model.

        NOTE: the displacement field won't work in slicer correctly if the correct itent code is set - the original should be left.
        """

        super().__init__("SyNANTs",
                         configuration_path,
                         dataloader)

    def _register(self, fixed_image_path: Path, moving_image_path: Path) -> None:
        """
        Wrapper around ants to register.
        """

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        # check that both images exist
        assert self.path_fixed.exists(
        ), f"File {self.path_fixed} does not exist."
        assert self.path_moving.exists(
        ), f"File {self.path_moving} does not exist."

        self.path_result_deformed, self.path_result_deformation = self._create_result_paths(self.path_fixed.stem,
                                                                                            self.path_moving.stem,
                                                                                            ".nii.gz",
                                                                                            ".nii.gz")

        # load boath images with ants
        fixed_image = ants.image_read(self.path_fixed.as_posix())
        moving_image = ants.image_read(self.path_moving.as_posix())

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

        deformed_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                               transformlist=registration['fwdtransforms'])
        self._save_results(torch.from_numpy(deformed_image.numpy(
        ).transpose(2, 1, 0)), Path(registration['invtransforms'][1]))
        # self._save_results(registration['warpedmovout'], registration['invtransforms'][1])

    def _save_results(self, deformed: torch.Tensor, deformation_path: Path) -> None:
        self.result_transformed_image_path, self.result_transformation_path = \
            self._create_result_paths(self.path_fixed.stem,
                                      self.path_moving.stem,
                                      ".nii.gz",
                                      ".nii.gz")

        # cope transformation
        shutil.copy(deformation_path, self.result_transformation_path)

        utils_nifti.set_intent_code(
            self.result_transformation_path, "NIFTI_INTENT_DISPVECT")

        # permute x and y
        deformed = deformed.permute(2, 1, 0)

        # save transformed image
        utils.save_image(
            deformed, self.result_transformed_image_path, self.dataloader.spacing)
