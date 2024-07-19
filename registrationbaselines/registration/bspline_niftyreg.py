from pathlib import Path
from tqdm import tqdm

from typing import List, Dict, Any

import wandb

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_niftyreg, utils_nifti
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.evaluation.evaluation import Evaluation


class BSplineNiftyReg(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:

        self.method_name = "BSplineNiftyReg"

        self.configuration = configuration

        base_dir = Path(__file__).parent.parent.absolute().parent
        self.path_reg_f3d = base_dir / Path(
            "registrationbaselines/libraries/NiftyReg/reg_f3d_ubuntu")

        # paths
        self.path_working_dir_path = Path()

        # command to call NiftyReg
        self.command: List[str] = []

    def register(self,
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
            utils_niftyreg.convert_control_point_grid_to_displacement_field(
                self.result_control_grid_path, self.path_fixed)

        # assign intent code to the displacement field
        utils_nifti.set_intent_code(
            self.path_result_deformation, "NIFTI_INTENT_DISPVECT")

    def _register_wandb_wrapper(self) -> None:
        """
        Register and evaluate all files and log to wand.

        @return: None
        """

        # IMPORTANT: this has to be called after creating wandb.agent()
        wandb.init(mode="offline")

        self.method_name = self.method_name + \
            f"_sim{wandb.config.similarity_metric.replace('-', '').replace(' ', '_')}"

        self._create_result_directories()

        assert len(self.dataloader) > 0, "Dataloader is empty."
        for item in tqdm(self.dataloader):
            break
            self.register(item["fixed_image"], item["moving_image"])

            # evaluate
        loader_transformations = data_loaders.BaselineTransformations(
            Path(wandb.config.result_path) / self.method_name)

        print("\nevaluate...")
        evaluation = Evaluation(
            Path(wandb.config.result_path), self.method_name)
        evaluation.evaluate(
            loader_transformations, self.dataloader)

        # print("\nplot...")
        # evaluation.visualize(
        #     loader_transformations, self.dataloader)

        print("\nlog to wandb...")
        evaluation.wandb_log()

    def __create_registration_command_list(self):
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
                                                                      wandb.config,
                                                                      only_value=True)

    def __outputs_exist(self):
        """
        We need this because it's not clear that blockmatching returns non-zero
        when failed
        """
        if self.path_result_deformed.exists() and self.path_result_deformation.exists():
            return True

        return False
