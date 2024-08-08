from abc import ABC, abstractmethod
from pathlib import Path

from typing import Dict, Any, Union

import yaml
import torch
import wandb
from tqdm import tqdm

from registrationbaselines.core import utils
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.evaluation.evaluation import Evaluation


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    method_name: str = ""

    configuration: Dict[str, Union[str, int, float, bool]] = {}

    dataloader: data_loaders.GenericDataset

    path_fixed: Path = Path()
    path_moving: Path = Path()

    path_results: Path = Path()
    path_dir_deformed: Path = Path()
    path_dir_deformations: Path = Path()
    method_dir: Path = Path()

    path_result_deformation: Path = Path()
    path_result_deformed: Path = Path()

    use_wandb: bool = False

    evaluator: Evaluation

    @abstractmethod
    def __init__(self,
                 configuration: Dict[str, Any],
                 dataloader: data_loaders.GenericDataset,
                 use_wandb: bool):
        """
        Initialize the registration model.
        """

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

    def register_dataset(self) -> None:
        """
        Register and evaluate all files and log to wand.

        @return: None
        """

        if self.use_wandb is False:
            self.configuration = self.convert_to_non_wandb_config(
                self.configuration)
            result_path = Path(self.configuration["result_path"])
        else:
            result_path = Path(
                self.configuration["parameters"]["result_path"]["values"][0])

        self._create_result_directories(self.method_name)

        for item in tqdm(self.dataloader):
            self._register(item["fixed_image"], item["moving_image"])

        loader_transformations = data_loaders.BaselineTransformations(
            self.method_dir)

        self.evaluator = Evaluation(result_path,
                                    self.method_dir.name,
                                    self.dataloader,
                                    loader_transformations)

        self.evaluator.evaluate()

        self.evaluator.visualize()

    def _perform_wandb_run(self) -> None:
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
        # IMPORTANT: this has to be called after creating wandb.agent()
        wandb.init(mode="online")

        buffer_ori_name = self.method_name

        self.method_name = self.create_method_name_for_wandb(wandb.config)

        self.register_dataset()

        self.evaluator.wandb_log()

        self.method_name = buffer_ori_name

    def perform_wandb_sweep(self) -> None:
        """
        Register all parameter sets.
        """

        self.use_wandb = True

        self.sweep_id = wandb.sweep(self.configuration,
                                    entity=None,
                                    project="reg_baselines")

        wandb.agent(self.sweep_id,
                    function=lambda: self._perform_wandb_run(),
                    entity=None,
                    project="reg_baselines")

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

        self.path_results = Path(self.configuration["result_path"]) if not self.use_wandb else Path(
            self.configuration["parameters"]["result_path"]["values"][0])

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
    def convert_to_non_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the config without the wandb config.
        """

        config = config["parameters"]

        new_config: Dict[str, str] = {}

        for key, value in config.items():
            new_config[key] = value["values"][0]

        return new_config

    def create_method_name_for_wandb(self, wandb_config: Dict[str, Union[str, int, float, bool]]) -> str:
        """
        Create the method name for wandb.
        """

        method_name = self.method_name

        for key, value in wandb_config.items():

            if key not in ['result_path', 'method_name']:
                if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
                    method_name += f"___{key}_{str(value).lower()}"
                else:
                    beautified_param = value.replace('-', '').replace(' ', '_')
                    method_name += f"___{key}_{beautified_param}"

        return method_name

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
