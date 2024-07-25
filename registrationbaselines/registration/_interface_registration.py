from abc import ABC, abstractmethod
from pathlib import Path

from typing import Dict, Any

import yaml
import torch
import wandb

from registrationbaselines.core import utils
from registrationbaselines.data_loading.data_loaders import GenericDataset


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    method_name: str = ""

    configuration: Dict[str, Any] = {}

    dataloader: GenericDataset

    path_fixed: Path = Path()
    path_moving: Path = Path()

    path_results: Path = Path()
    path_dir_deformed: Path = Path()
    path_dir_deformations: Path = Path()
    method_dir: Path = Path()

    path_result_deformation: Path = Path()
    path_result_deformed: Path = Path()

    @abstractmethod
    def __init__(self,
                 configuration: Dict[str, Any],
                 dataloader: GenericDataset):
        """
        Initialize the registration model.
        """

    @abstractmethod
    def register(self,
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

    @abstractmethod
    def _register_wandb_wrapper(self) -> None:
        """
        This wraps register() and is used by wandb.agent.
        This has to (in order)
            0. initialise wandb with wandb.init()
            1. create a unique method name for the current run
            2. loop over the entire dataset and call register() for each item.
            3. create a BaselineTransformations loader
                - the path should be result_path/method_name
            4. create an Evaluation object
                - initialise with result_path and method_name
            5. call evaluate()
            6. call visualis() [optional]
            7. call wandb_log() to log evaluation metrics (results)

            WARNING
            wandb.config doesn't reflect the entire config file,
            just the config for the current run
        """

    def register_all_parametr_sets(self) -> None:
        """
        Register all parameter sets.
        """

        self.sweep_id = wandb.sweep(self.configuration,
                                    entity=None,
                                    project="reg_baselines")

        wandb.agent(self.sweep_id,
                    function=lambda: self._register_wandb_wrapper(),
                    entity=None,
                    project="reg_baselines",
                    count=None)

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
        utils.save_image(deformed,
                         self.path_result_deformed,
                         self.dataloader.spacing)

        # SAVE DEFORMATION
        utils.save_image(deformation,
                         self.path_result_deformation,
                         self.dataloader.spacing)

        if not self.path_result_deformed.exists():
            raise FileNotFoundError(
                f"File {self.path_result_deformed} couldn't be saved.")
        if not self.path_result_deformation.exists():
            raise FileNotFoundError(
                f"File {self.path_result_deformation} couldn't be saved.")

    def _create_result_directories(self, method_directory_name: str):
        """
        Create the directories to save the results.
        """

        self.path_results = Path(
            self.configuration["parameters"]["result_path"]["value"])

        # create directory in base_dir called method
        self.method_dir = self.path_results / \
            self.dataloader.name / method_directory_name
        self.method_dir.mkdir(parents=True, exist_ok=True)

        # create two subdirectories 'deformed' and 'deformations'
        self.path_dir_deformed = self.method_dir / 'deformed'
        self.path_dir_deformed.mkdir(parents=True, exist_ok=True)

        self.path_dir_deformations = self.method_dir / 'deformations'
        self.path_dir_deformations.mkdir(parents=True, exist_ok=True)

        if not self.path_dir_deformations.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_deformations} couldn't be created.")
        if not self.path_dir_deformed.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_deformed} couldn't be created.")

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

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
