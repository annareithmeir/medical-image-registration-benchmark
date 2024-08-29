from pathlib import Path
import shutil

from abc import abstractmethod
from typing import Optional

import torch
import wandb
from tqdm import tqdm

from registrationbaselines.core import utils_wandb
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.interfaces import _interface_core
from registrationbaselines.io import save


class RegistrationInterface(_interface_core.InterfaceCore):
    """
    Abstract base class for registration models.
    """

    dataloader: data_loaders.GenericDataset

    path_fixed: Path = Path()
    path_moving: Path = Path()

    path_dir_deformed: Path = Path()
    path_dir_deformations: Path = Path()

    path_result_deformation: Path = Path()
    path_result_deformed: Path = Path()

    evaluator: Evaluation
    use_masked_evaluation: bool

    def __init__(self,
                 method_name: str,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True) -> None:
        """
        Initialize the registration model.
        """

        super().__init__(method_name,
                         configuration_path,
                         dataloader.name)

        self.dataloader = dataloader
        self.use_masked_evaluation = use_masked_evaluation

    @abstractmethod
    def _register(self,
                  fixed_image_path: Path,
                  moving_image_path: Path) -> None:
        """
        Register moving_image to fixed_image.

        This function has to call _save_results() at the end.

        @param fixed_image_path: path to the fixed image
        @type fixed_image_path: Path

        @param moving_image_path: path to the moving image
        @type moving_image_path: Path

        @return: None
        """

    def execute_with_one_parameter_set(self) -> None:
        """
        Register and evaluate all files and log to wand.

        @return: None
        """

        if "model_path" in self.general_configuration["parameters"] and self.use_wandb:
            wandb.finish()
            raise ValueError(
                "DL mode can't be used with wandb sweeps for registration, because the sweep was done in training.")

        self._create_run_parameters()

        self._create_run_directory()

        self._save_run_configuration()

        for item in tqdm(self.dataloader):
            self._register(item["fixed_image"], item["moving_image"])

        loader_transformations = data_loaders.BaselineTransformations(
            self.path_dir_run)

        self.evaluator = Evaluation(run_path=self.path_dir_run,
                                    dataset_data=self.dataloader,
                                    dataset_transformations=loader_transformations,
                                    use_zero_displacement=False,
                                    use_masked_evaluation=self.use_masked_evaluation)

        self.evaluator.evaluate()

        self.evaluator.visualize()

        if self.use_wandb:
            self.evaluator.wandb_log()

    def evaluate_with_zero_displacement(self) -> None:
        """
        Evaluate with zero displacement.

        @return: None
        """

        self.run_name = "_zeroDisplacement"

        if self.use_wandb:
            self.run_configuration = wandb.config
        else:
            self.run_configuration = utils_wandb.convert_to_non_wandb_config(
                self.general_configuration)

        self._create_run_directory()
        self.path_dir_deformations.rmdir()
        self.path_dir_deformed.rmdir()

        self.evaluator = Evaluation(run_path=self.path_dir_run,
                                    dataset_data=self.dataloader,
                                    dataset_transformations=None,
                                    use_zero_displacement=True,
                                    use_masked_evaluation=self.use_masked_evaluation)

        # if a zero displacement evaluation is already present, raise
        if (self.path_dir_dataset / self.path_dir_run.name).exists():
            raise FileExistsError(
                f"Directory {self.path_dir_run} already exists.")

        self.evaluator.evaluate()

        self.evaluator.visualize()

        # move the results directory one level up
        shutil.move(self.path_dir_run, self.path_dir_dataset)

        # if the path_dir_method_is empty, delete it
        if not list(self.path_dir_method.iterdir()):
            self.path_dir_method.rmdir()

    def get_transformation_path(self):
        """
        Return the transformation model.
        """
        return self.path_result_deformation

    def get_transformed_image_path(self):
        """
        Return the transformed image.
        """
        return self.path_result_deformed

    def _save_results(self, deformed: torch.Tensor, deformation: torch.Tensor):
        """
        Save the results of the registration.
        """

        self.path_result_deformed, \
            self.path_result_deformation = self._create_result_paths(self.path_fixed.stem,
                                                                     self.path_moving.stem,
                                                                     ".nii.gz",
                                                                     ".nii.gz")

        # SAVE DEFORMED IMAGE
        save.save_image(deformed,
                                               self.path_result_deformed,
                                               self.dataloader.spacing)

        # SAVE DEFORMATION
        save.save_displacement(deformation,
                                                      self.path_result_deformation,
                                                      self.dataloader.spacing + (1,))

        if not self.path_result_deformed.exists():
            raise FileNotFoundError(
                f"File {self.path_result_deformed} couldn't be saved.")
        if not self.path_result_deformation.exists():
            raise FileNotFoundError(
                f"File {self.path_result_deformation} couldn't be saved.")

    def _create_result_paths(self,
                             name_fixed: str,
                             name_moving: str,
                             extension_image: str,
                             extension_transformation: str):
        """
        Create the paths for the result files (warped image and transformation).
        """

        name_moving = name_moving.replace(".nii", "")
        name_fixed = name_fixed.replace(".nii", "")

        name_moving = name_moving.replace(".gz", "")
        name_fixed = name_fixed.replace(".gz", "")

        path_dir_deformed = self.path_dir_deformed / \
            f"{name_moving}_deformed_to_{name_fixed}"
        path_deformation = self.path_dir_deformations / \
            f"{name_moving}_deformation_to_{name_fixed}"

        path_dir_deformed = Path(
            path_dir_deformed.as_posix() + extension_image)
        path_deformation = Path(
            path_deformation.as_posix() + extension_transformation)

        return Path(path_dir_deformed), Path(path_deformation)

    def _create_run_directory(self) -> None:
        """
        Create the run directory in the method directory.
        """

        # if we are in DL mode, we want to copy the run name from the model path
        if "model_path" in self.run_configuration:
            self.run_name = Path(
                self.run_configuration["model_path"]).parent.name

        self.path_dir_run = self.path_dir_method / self.run_name
        self.path_dir_run.mkdir(parents=True, exist_ok=True)

        # create two subdirectories 'deformed' and 'deformations'
        self.path_dir_deformed = self.path_dir_run / 'deformed'
        self.path_dir_deformed.mkdir(parents=True, exist_ok=True)

        self.path_dir_deformations = self.path_dir_run / 'deformations'
        self.path_dir_deformations.mkdir(parents=True, exist_ok=True)

        if not self.path_dir_deformations.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_deformations} couldn't be created.")
        if not self.path_dir_deformed.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_deformed} couldn't be created.")
