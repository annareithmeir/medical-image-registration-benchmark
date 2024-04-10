from abc import ABC, abstractmethod
from pathlib import Path


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    @abstractmethod
    def __init__(self, configuration):
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
