from abc import ABC, abstractmethod
from pathlib import Path


class TrainingInterface(ABC):
    """
    Abstract base class for training procedure.
    """

    @abstractmethod
    def __init__(self, configuration):
        """
        Initialize the training procedure.
        """

        self.initial_weights_path = None
        self.train_data_path = None

    @abstractmethod
    def train(self, print_progress: bool = False):
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
