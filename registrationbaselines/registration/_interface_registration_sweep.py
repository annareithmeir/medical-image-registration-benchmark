from abc import ABC, abstractmethod
from pathlib import Path
from functools import partial

from typing import Dict, Any

import yaml
import wandb
from torch.utils.data import Dataset


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    configuration_register: Dict[str, Any]
    configuration_wandb: Dict[str, Any]

    sweep_id: str

    method: str

    path_results: Path
    path_deformed: Path
    path_deformations: Path

    result_transformation_path: Path
    result_transformed_image_path: Path

    dataloader: Dataset

    @abstractmethod
    def __init__(self, configuration: Dict[str, Any],
                 configuration_register: Dict[str, Any]):
        """
        Initialize the registration model.
        """

    @abstractmethod
    def register(self,
                 fixed_image_path: Path,
                 moving_image_path: Path,
                 print_progress: bool = False):
        """
        Register moving_image to fixed_image.
        """

    @abstractmethod
    def _register_wandb_wrapper(self, method_name: str):
        """
        This wraps register() and is used by wandb.agent.
        This has to loop over the entire dataset and call register() for each item.
        """

    def register_all_parametr_sets(self, dataloader: Dataset):
        """
        Register all parameter sets.
        """
        self.dataloader = dataloader

        self.sweep_id = wandb.sweep(self.configuration_wandb,
                                    entity=None,
                                    project="reg_baselines")

        wandb.agent(self.sweep_id,
                    # use partial to bind self to the method
                    function=lambda: self._register_wandb_wrapper(
                        self.configuration_wandb["name"]),
                    entity=None,
                    project="reg_baselines",
                    count=None)

    def get_transformation_path(self):
        """
        Return the transformation model.
        """
        return self.result_transformation_path

    def get_transformed_image_path(self):
        """
        Return the transformed image.
        """
        return self.result_transformed_image_path

    @abstractmethod
    def _save_results(self, deformed, deformation):
        """
        Save the results of the registration.
        """

    def _create_result_directories(self):
        """
        Create the directories to save the results.
        """

        self.path_results = Path(
            self.configuration_wandb["parameters"]["result_path"]["value"])

        # create directory in base_dir called method
        method_dir = self.path_results / self.method
        method_dir.mkdir(parents=True, exist_ok=True)

        # create two subdirectories 'deformed' and 'deformations'
        self.path_deformed = method_dir / 'deformed'
        self.path_deformed.mkdir(parents=True, exist_ok=True)

        self.path_deformations = method_dir / 'deformations'
        self.path_deformations.mkdir(parents=True, exist_ok=True)

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

        path_deformed = self.path_deformed / \
            f"{name_moving}_deformed_to_{name_fixed}"
        path_deformation = self.path_deformations / \
            f"{name_moving}_deformation_to_{name_fixed}"

        path_deformed = Path(
            path_deformed.as_posix() + extension_image)
        path_deformation = Path(
            path_deformation.as_posix() + extension_transformation)

        return Path(path_deformed), Path(path_deformation)

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
