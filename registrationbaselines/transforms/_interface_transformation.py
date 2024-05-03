from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class TransformationInterface(ABC):
    """
    Abstract base class for transformations.
    """

    configuration = None

    method = None

    path_results = None
    path_deformed = None

    @abstractmethod
    def __init__(self, configuration):
        """
        Initialize the transformation.
        """

    @abstractmethod
    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Apply a transformation to an image using the provided transformation.
        """

    def __save_results(self, deformed):
        """
        Save the results of the registration.
        """

    def __create_result_directories(self):
        """
        Create the directory to save the results.
        """

        self.path_results = Path(
            self.configuration["result_path"]) / self.method

        # create directory in base_dir called method
        method_dir = self.path_results / self.method
        method_dir.mkdir(parents=True, exist_ok=True)

        # create subdirectory 'deformed'
        self.path_deformed = method_dir / 'deformed'
        self.path_deformed.mkdir(parents=True, exist_ok=True)

    def __create_result_path(self,
                             name_fixed: str,
                             name_moving: str,
                             extension_image: str):
        """
        Create the paths for the result files (warped image and transformation).
        """

        path_deformed = self.path_deformed / \
            f"{name_moving}_deformed_to_{name_fixed}"

        path_deformed = path_deformed.resolve().as_posix() + extension_image

        return Path(path_deformed)

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
