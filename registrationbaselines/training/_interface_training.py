from pathlib import Path
import os

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Any

import torch
import wandb

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.core import utils


class TrainingInterface(ABC):
    """
    Abstract base class for training procedure.
    """

    method_name: str

    configuration: Dict[str, Union[str, int, float, bool]]

    train_dataset: data_loaders.GenericDataset
    val_dataset: Optional[data_loaders.GenericDataset] = None

    model: Any

    initial_weights_path: Path
    train_data_path: Path
    base_dir: Path
    model_dir: Path

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

        buffer_ori_config = self.configuration

        if self.use_wandb:
            self.configuration = wandb.config
        else:
            self.configuration = utils.convert_to_non_wandb_config(
                self.configuration)

        result_path = Path(self.configuration["result_path"])

        # todo
        # self._create_result_directories(self.method_name)

        # prepare model folder
        self.model_dir = self.base_dir / result_path
        os.makedirs(self.model_dir, exist_ok=True)

        self._train()

        self.configuration = buffer_ori_config

    def _perform_wandb_run(self) -> None:
        """
        Perform a single run with wandb.
        """

        # IMPORTANT: this has to be called after creating wandb.agent()
        wandb.init()

        # todo use run names
        buffer_ori_name = self.method_name
        self.method_name = utils.create_method_name_for_wandb(self.method_name,
                                                              wandb.config)

        self.train_with_one_parameter_set()

        self.method_name = buffer_ori_name

    def perform_wandb_sweep(self) -> None:
        """
        Train with all parameter sets.
        """

        self.use_wandb = True

        self.sweep_id = wandb.sweep(self.configuration,
                                    entity=None,
                                    project="reg_baselines")

        self.agent = wandb.agent(self.sweep_id,
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
