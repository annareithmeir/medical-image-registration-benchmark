from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import os
import torch

class TrainingInterface(ABC):
    """
    Abstract base class for training procedure.
    """

    @abstractmethod
    def __init__(self, config_path):
        """
        Initialize the training procedure.
        """

        self.initial_weights_path = None
        self.train_data_path = None
        self.model = None
        self.base_dir = None
        self.configuration = None

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

    @abstractmethod
    def train(self):
        """
        Train with given training data.
        """

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
        torch.save(self.model.state_dict(), self.base_dir / self.configuration['initial_weights_path']) # '.pth'

    # @staticmethod
    # def read_config(file_path: Path):
    #     """
    #     Read the configuration file.
    #     """
    #
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         return yaml.safe_load(file)

