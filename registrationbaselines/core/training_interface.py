from abc import ABC, abstractmethod
from pathlib import Path
import yaml


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

    @abstractmethod
    def train(self):
        """
        Train with given training data.
        """

    @abstractmethod
    def get_trained_model_path(self):
        """
        Return the trained model.
        """

    @abstractmethod
    def get_initial_weights_path(self):
        """
        Return the initial weights that have been used for training (for reproducibility).
        """

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

