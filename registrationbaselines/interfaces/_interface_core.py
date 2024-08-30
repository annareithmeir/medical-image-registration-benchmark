import os
from pathlib import Path
import json
import uuid

from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional

import wandb
import wandb.sdk

from registrationbaselines.io import load
from registrationbaselines.core import utils_wandb


class InterfaceCore(ABC):
    """
    Abstract base class for registration models.
    """

    method_name: str
    run_name: str

    general_configuration: Dict[str, List[Union[str, int, float, bool]]]
    run_configuration: Dict[str, Union[str, int, float, bool]]

    path_dir_results: Path
    path_dir_dataset: Path
    path_dir_method: Path
    path_dir_run: Path

    use_wandb: bool = False

    model_path: Optional[Path] = None

    def __init__(self,
                 method_name: str,
                 configuration_path: Path,
                 dataset_name: str) -> None:
        """
        Initialize the registration model.
        """

        self.method_name = method_name

        self.general_configuration = load.read_config(configuration_path)

        self.base_dir = Path(__file__).parent.parent.absolute().parent

        self._created_method_directory(dataset_name)

    def perform_wandb_sweep(self) -> None:
        """
        Train with all parameter sets.
        """

        self.use_wandb = True

        self.sweep_id = wandb.sweep(self.general_configuration,
                                    entity=None,
                                    project="reg_baselines")

        wandb.agent(self.sweep_id,
                    function=lambda: self._perform_wandb_run(),
                    entity=None,
                    project="reg_baselines")

    def _perform_wandb_run(self) -> None:
        """
        Perform a single run with wandb.
        """

        # IMPORTANT: this has to be called after creating wandb.agent()
        wandb.init()

        self.execute_with_one_parameter_set()

    @abstractmethod
    def execute_with_one_parameter_set(self) -> None:
        """
        Execute with one parameter set.
        """

    def _created_method_directory(self, dataset_name: str) -> None:
        """
        Create the method directory.
        """

        self.path_dir_results = Path(
            self.general_configuration["parameters"]["result_path"]["values"][0])

        # create directory in base_dir called method
        self.path_dir_dataset = self.path_dir_results / dataset_name
        self.path_dir_method = self.path_dir_dataset / self.method_name
        self.path_dir_method.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _create_run_directory(self) -> None:
        """
        Create the run directory.
        """

    def _create_run_parameters(self) -> None:
        """
        Creates the run config and the run name
        """

        if self.use_wandb:
            self.run_configuration = wandb.config
            self.run_name = self.method_name + \
                f"_{utils_wandb.get_current_wandb_run_name()}"
        else:
            self.run_configuration = utils_wandb.convert_to_non_wandb_config(
                self.general_configuration)
            # if we are in DL mode, we want to copy the run name from the model path
            if self.model_path is not None:
                self.run_name = self.model_path.parent.name
            else:
                self.run_name = self.method_name + f"_{uuid.uuid4()}"

    def _save_run_configuration(self) -> None:
        """
        Save the run configuration.
        """

        config_path = self.path_dir_run / "config.yaml"

        if isinstance(self.run_configuration, wandb.sdk.wandb_config.Config):
            config = self.run_configuration.as_dict()
        else:
            config = self.run_configuration

            # sort dict alphabetically by keys, to match wandb
            config = dict(sorted(config.items()))

        with open(config_path, "w") as file:
            json.dump(config, file, indent=4)

    def _handle_device_selection(self) -> str:
        """
        Handle device selection.
        If GPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to the selected GPU and return 'cuda'.
        If CPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to -1 and return 'cpu'.
        """

        num = str(self.general_configuration["parameters"]["gpu"]["values"][0])

        if num and (num != '-1'):
            device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = num
        else:
            device = 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        return device
