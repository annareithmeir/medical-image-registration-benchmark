from abc import ABC, abstractmethod
from pathlib import Path

from typing import Dict, Any, Union

import wandb


class TrainingInterface(ABC):
    """
    Abstract base class for training procedure.
    """

    configuration: Dict[str, Union[str, int, float, bool]] = {}

    @abstractmethod
    def __init__(self, config_path):
        """
        Initialize the training procedure.
        """

        self.initial_weights_path = None
        self.train_data_path = None
        self.model = None

    @abstractmethod
    def train(self, use_wandb: bool) -> None:
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

    def perform_wandb_training_sweep(self) -> None:
        """
        Train with all parameter sets.
        """

        self.use_wandb = True

        self.sweep_id = wandb.sweep(self.configuration,
                                    entity=None,
                                    project="reg_baselines")

        wandb.agent(self.sweep_id,
                    function=lambda: self.train(use_wandb=True),
                    entity=None,
                    project="reg_baselines")
