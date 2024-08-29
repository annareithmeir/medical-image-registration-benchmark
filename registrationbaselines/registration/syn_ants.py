import shutil
from pathlib import Path

import ants
import torch
import SimpleITK as sitk

from registrationbaselines.data_loading import data_loaders
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

    def _register(self, fixed_image_path: Path, moving_image_path: Path) -> None:
        """
        Wrapper around ants to register.

        BUGFIX 0: ants doesn't permute and flip on loading, so we have permute images on loading and permute and flip the displacement field on saving.
                  We also have to save the forward transform, as this was also used to register.
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

        deformed_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                               transformlist=registration['fwdtransforms'])
        # BUGFIX 0
        self._save_results(torch.from_numpy(deformed_image.numpy()),
                           Path(registration['fwdtransforms'][0]))

    def _save_results(self, deformed: torch.Tensor, deformation_path: Path) -> None:

        self.result_transformed_image_path, self.result_transformation_path = \
            self._create_result_paths(self.path_fixed.stem,
                                      self.path_moving.stem,
                                      ".nii.gz",
                                      ".nii.gz")

        # copy transformation
        shutil.copy(deformation_path, self.result_transformation_path)

        # BUGFIX 0
        sitk_im = sitk.ReadImage(self.result_transformation_path)
        sitk_tensor = torch.from_numpy(
            sitk.GetArrayFromImage(sitk_im)).unsqueeze(3)
        # todo this should be utils.reverse()
        sitk_tensor = sitk_tensor[..., [2, 1, 0]]
        sitk_tensor = sitk_tensor.permute(4, 3, 2, 1, 0)
        sitk_im = sitk.GetImageFromArray(sitk_tensor.cpu().numpy())
        sitk.WriteImage(sitk_im, self.result_transformation_path)

        utils_nifti.set_intent_code(
            self.result_transformation_path, "NIFTI_INTENT_DISPVECT")

        # save transformed image
        save.save_image(
            deformed, self.result_transformed_image_path, self.dataloader.spacing)
