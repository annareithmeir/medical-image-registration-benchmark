from pathlib import Path
import os
import uuid

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Any

import torch
import wandb

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.interfaces import _interface_core


class TrainingInterface(_interface_core.InterfaceCore):
    """
    Abstract base class for training procedure.
    """

    # todo init wandb with project, group and name

    train_dataset: data_loaders.GenericDataset
    val_dataset: Optional[data_loaders.GenericDataset] = None

    model: Any

    initial_weights_path: Path
    train_data_path: Path
    base_dir: Path
    path_dir_train_results: Path

    device: str

    sweep_id: str = ""

    def __init__(self,
                 method_name: str,
                 configuration_path: Path,
                 path_results: Path,
                 train_dataset: data_loaders.GenericDataset,
                 val_dataset: Optional[data_loaders.GenericDataset] = None,
                 use_logger=False) -> None:
        """
        Initialize the training procedure.
        """
        super().__init__(method_name,
                         configuration_path,
                         path_results,
                         train_dataset.name,
                         use_logger=use_logger)

        self._create_train_results_directory()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if (len(self.train_dataset) == 0) or (self.val_dataset is not None and len(self.val_dataset) == 0):
            raise ValueError("The dataset is empty.")

    @abstractmethod
    def _train(self) -> None:
        """
        Train with given training data.
        """

    def execute_with_one_parameter_set(self) -> None:
        """
        Train with one parameter set.
        """

        self._create_run_parameters()

        self._create_run_directory()

        self._save_run_configuration()

        self._train()

    def _create_train_results_directory(self) -> None:
        """
        Create the directories to save the results.
        """

        # create 'train' subdirectory
        self.path_dir_train_results = self.path_dir_method / 'train'
        self.path_dir_train_results.mkdir(parents=True, exist_ok=True)

        if not self.path_dir_train_results.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_train_results} couldn't be created.")

    def _create_run_directory(self) -> None:
        """
        Create the run directory in the train results directory.
        """
        self.path_dir_run = self.path_dir_train_results / self.run_name
        self.path_dir_run.mkdir(parents=True, exist_ok=True)

    def save_initial_weights(self):
        assert self.model is not None, "Model is not yet initialized!"
        self.model.save(self.get_initial_weights_path())

    def get_trained_model_path(self) -> Path:
        return self.path_dir_run / f"model_epoch{self.run_configuration['epochs']:05d}_final.pt"

    def get_initial_weights_path(self) -> Path:
        return self.path_dir_run / "model_epoch00000_initial.pt"
