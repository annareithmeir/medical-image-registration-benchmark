from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    configuration = None

    method = None

    path_results = None
    path_deformed = None
    path_deformations = None

    @abstractmethod
    def __init__(self, configuration_path: Path):
        """
        Initialize the registration model.
        """

    @abstractmethod
    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        """
        Register moving_image to fixed_image.
        """

    @abstractmethod
    def get_transformation_path(self):
        """
        Return the transformation model.
        """

    @abstractmethod
    def get_transformed_image_path(self):
        """
        Return the transformed image.
        """

    @abstractmethod
    def __save_results(self, deformed, deformation):
        """
        Save the results of the registration.
        """

    def __create_result_directories(self):
        """
        Create the directories to save the results.
        """

        self.path_results = Path(
            self.configuration["result_path"]) / self.method

        # create directory in base_dir called method
        method_dir = self.path_results / self.method
        method_dir.mkdir(parents=True, exist_ok=True)

        # create two subdirectories 'deformed' and 'deformations'
        self.path_deformed = method_dir / 'deformed'
        self.path_deformed.mkdir(parents=True, exist_ok=True)

        self.path_deformations = method_dir / 'deformations'
        self.path_deformations.mkdir(parents=True, exist_ok=True)

    def __create_result_paths(self,
                              name_fixed: str,
                              name_moving: str,
                              extension_image: str,
                              extension_transformation: str):
        """
        Create the paths for the result files (warped image and transformation).
        """

        path_deformed = self.path_deformed / \
            f"{name_moving}_deformed_to_{name_fixed}"
        path_deformation = self.path_deformations / \
            f"{name_moving}_deformation_to_{name_fixed}"

        path_deformed = path_deformed.resolve().as_posix() + extension_image
        path_deformation = path_deformation.resolve().as_posix() + extension_transformation  # nopep8

        return Path(path_deformed), Path(path_deformation)

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
