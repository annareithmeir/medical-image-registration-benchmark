import ants
from pathlib import Path
import wandb
from tqdm import tqdm
from typing import Dict, Any
import torch

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg, utils_nifti, utils
import SimpleITK as sitk

class SyNANTs(RegistrationInterface):
    """
    SyN registration using ANTs.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self,
                 path_configuration: Path,
                 dataloader: data_loaders.GenericDataset) -> None:
        """
        Initialize the registration model.

        NOTE: the displacement field won't work in slicer correctly if the correct itent code is set - the original should be left.
        """

        self.method_name = "SyNANTs"

        self.configuration = utils.read_config(path_configuration)

        self.dataloader = dataloader

        self.base_dir = Path(__file__).parent.parent.absolute().parent

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

        if self.use_wandb is False:
            grad_step = self.configuration["grad_step"]
            flow_sigma = self.configuration["flow_sigma"]
            total_sigma = self.configuration["total_sigma"]
        else:
            grad_step = wandb.config["grad_step"]
            flow_sigma = wandb.config["flow_sigma"]
            total_sigma = wandb.config["total_sigma"]

        # Perform registration
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            grad_step=grad_step,
            flow_sigma=flow_sigma,
            total_sigma= total_sigma,
            type_of_transform='SyNOnly',
            write_composite_transform=True  # nopep8 this outputs one .h5 transform, otherwise we have a .nii.gz and .mat
        )

        #
        #deformed_image = registration['warpedmovout']
        # print(registration['fwdtransforms'])
        #deformation = ants.read_transform(registration['invtransforms']).numpy()

        # deformation = ants.create_warped_grid(deformation, grid_spacing=(1,1,1), fixed_reference_image=fixed_image)

        # self._save_results(torch.tensor(deformed_image.numpy()), torch.tensor(deformation.numpy()))
        self._save_results(registration['warpedmovout'], registration['invtransforms'])

    def _save_results(self, deformed, deformation):
        self.result_transformed_image_path, self.result_transformation_path = \
            self._create_result_paths(self.path_fixed.stem,
                                      self.path_moving.stem,
                                      ".nii.gz",
                                      ".nii.gz")

        # save transformation (by converting to .nii.gz)
        utils_nifti.convert_h5_to_nii(self.path_fixed,
                                      Path(deformation),
                                      self.result_transformation_path)
        utils_nifti.set_intent_code(self.result_transformation_path, "NIFTI_INTENT_DISPVECT")

        displacement_sitk = sitk.ReadImage(self.result_transformation_path)
        displacement_sitk.SetSpacing((1,1,*self.dataloader.spacing))
        displacement_array = sitk.GetArrayFromImage(
            displacement_sitk)
        displacement_array= displacement_array[..., [2, 1, 0]]
        displacement_sitk = sitk.GetImageFromArray(displacement_array, isVector=True)
        sitk.WriteImage(displacement_sitk, self.result_transformation_path)

        # save transformed image
        deformed.to_filename(self.result_transformed_image_path)

