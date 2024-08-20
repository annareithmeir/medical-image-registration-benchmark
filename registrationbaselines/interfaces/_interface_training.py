from pathlib import Path
import os
import uuid

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Any

import torch
import wandb

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.core import utils, utils_wandb


class TrainingInterface(ABC):
    """
    Abstract base class for training procedure.
    """

    method_name: str
    run_name: str

    general_configuration: Dict[str, Union[str, int, float, bool]]
    run_configuration: Dict[str, Union[str, int, float, bool]]

    train_dataset: data_loaders.GenericDataset
    val_dataset: Optional[data_loaders.GenericDataset] = None

    model: Any

    initial_weights_path: Path
    train_data_path: Path
    base_dir: Path
    method_dir: Path
    path_dir_train_results: Path

    sweep_id: str = ""

    use_wandb: bool = False

    def __init__(self,
                 method_name: str,
                 configuration_path: Path,
                 train_dataset: data_loaders.GenericDataset,
                 val_dataset: Optional[data_loaders.GenericDataset] = None) -> None:
        """
        Initialize the training procedure.
        """

        self.method_name = method_name

        self.configuration = utils.read_config(configuration_path)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.base_dir = Path(__file__).parent.parent.absolute().parent

    @abstractmethod
    def _train(self) -> None:
        """
        Train with given training data.
        """

    def train_with_one_parameter_set(self) -> None:
        """
        Train with one parameter set.
        """

        if self.use_wandb:
            self.run_configuration = wandb.config
            self.run_name = self.method_name + \
                f"_{utils_wandb.get_current_wandb_run_name()}"
        else:
            self.run_configuration = utils_wandb.convert_to_non_wandb_config(
                self.configuration)
            self.run_name = self.method_name + f"_{uuid.uuid4()}"

        self.run_directory = self.path_dir_train_results / self.run_name
        self.run_directory.mkdir(parents=True, exist_ok=True)

        self._train()

    def _perform_wandb_run(self) -> None:
        """
        Perform a single run with wandb.
        """

        # IMPORTANT: this has to be called after creating wandb.agent()
        wandb.init()

        self.train_with_one_parameter_set()

    def perform_wandb_sweep(self) -> None:
        """
        Train with all parameter sets.
        """
        self._create_result_directories()

        self.use_wandb = True

        self.sweep_id = wandb.sweep(self.configuration,
                                    entity=None,
                                    project="reg_baselines")

        wandb.agent(self.sweep_id,
                    function=lambda: self._perform_wandb_run(),
                    entity=None,
                    project="reg_baselines")

    def get_trained_model_path(self) -> Path:
        """
        Return the trained model.
        """
        return self.configuration['result_model_path']

    def get_initial_weights_path(self) -> Path:
        """
        Return the initial weights that have been used for training (for reproducibility).
        """
        return self.configuration['initial_weights_path']

    def _save_initial_weights(self) -> None:
        """
        Save the model's initial weights for better reproducibility
        TODO this is not yet used in the code
        @return:
        """
        assert self.model is not None, "Model is not yet initialized!"
        torch.save(self.model.state_dict(), self.base_dir /
                   self.configuration['initial_weights_path'])  # '.pth'

    def _create_result_model_path(self, result_model_path: Path) -> None:
        """
        Creates directory where the trained model is saved
        @param result_model_path: directory path
        @return:
        """
        if not os.path.isdir(result_model_path):
            os.mkdir(result_model_path)

    def _handle_device_selection(self) -> str:
        """
        Handle device selection.
        If GPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to the selected GPU and return 'cuda'.
        If CPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to -1 and return 'cpu'.
        """

        num = self.configuration['gpu']

        if num and (num != '-1'):
            device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = num
        else:
            device = 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        return device

    def _create_result_directories(self) -> None:
        """
        Create the directories to save the results.
        """

        self.path_results = Path(
            self.configuration["parameters"]["result_path"]["values"][0])

        # create directory in base_dir called method
        self.method_dir = self.path_results / self.train_dataset.name / self.method_name
        self.method_dir.mkdir(parents=True, exist_ok=True)

        # create 'train' subdirectory
        self.path_dir_train_results = self.method_dir / 'train'
        self.path_dir_train_results.mkdir(parents=True, exist_ok=True)

        if not self.path_dir_train_results.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_train_results} couldn't be created.")
    def save_initial_weights(self):
        assert self.model is not None, "Model is not yet initialized!"
        self.model.save(self.get_initial_weights_path())

    def get_trained_model_path(self) -> Path:
        return self.path_dir_run / f"model_epoch{self.run_configuration['epochs']:05d}_final.pt"

    def get_initial_weights_path(self) -> Path:
        return self.path_dir_run / "model_epoch00000_initial.pt"
