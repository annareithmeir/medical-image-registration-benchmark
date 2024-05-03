from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    configuration = None

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

    def __save_results(self, deformed, deformation):
        """
        Save the results of the registration.
        """

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
