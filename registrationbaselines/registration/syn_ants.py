import ants
from pathlib import Path
import wandb
from tqdm import tqdm
from typing import Dict, Any
import torch

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_nifti


class SyNANTs(RegistrationInterface):
    """
    SyN registration using ANTs.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self,
                 configuration: Dict[str, Any],
                 dataloader: data_loaders.GenericDataset) -> None:
        """
        Initialize the registration model.

        NOTE: the displacement field won't work in slicer correctly if the correct itent code is set - the original should be left.
        """

        self.method_name = "SyNANTs"
        self.method_name_ori = self.method_name

        self.base_dir = Path(__file__).parent.parent.absolute().parent

        self.configuration = configuration
        self.dataloader = dataloader

        self._create_result_directories(self.method_name)

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False, sweep: bool =False):
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

        if sweep:
            grad_step = wandb.config.grad_step
            flow_sigma = wandb.config.flow_sigma
            total_sigma = wandb.config.total_sigma
        else:
            grad_step = self.configuration["parameters"]["grad_step"]["values"]
            flow_sigma = self.configuration["parameters"]["flow_sigma"]["values"]
            total_sigma = self.configuration["parameters"]["total_sigma"]["values"]

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

        self._save_results(
            registration['warpedmovout'], registration['fwdtransforms'])

    def _register_wandb_wrapper(self) -> None:
        """
        Register and evaluate all files and log to wand.

        @return: None
        """

        # IMPORTANT: this has to be called after creating wandb.agent()
        wandb.init(mode="online")

        self.method_name = self.method_name_ori + \
                           f"_gradstep{wandb.config.grad_step}"+f"_flowsigma{wandb.config.flow_sigma}"

        self._create_result_directories(self.method_name)

        assert len(self.dataloader) > 0, "Dataloader is empty."
        for item in tqdm(self.dataloader):
            self.register(item["fixed_image"], item["moving_image"], sweep = True)
        #
        #     # evaluate
        print(self.method_name)
        print(Path(wandb.config.result_path) / self.dataloader.name / self.method_name)
        loader_transformations = data_loaders.BaselineTransformations(
            Path(wandb.config.result_path) / self.dataloader.name / self.method_name)
        #
        print("\nevaluate...")
        evaluation = Evaluation(Path(wandb.config.result_path),
                                self.method_dir.name,
                                self.dataloader,
                                loader_transformations)
        evaluation.evaluate()

        print("\nplot...")
        evaluation.visualize()

        print("\nlog to wandb...")
        evaluation.wandb_log()

        wandb.finish()

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

        # save transformed image
        deformed.to_filename(self.result_transformed_image_path)

