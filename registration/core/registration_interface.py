from abc import ABC, abstractmethod

from .configurations import RegistrationConfiguration


class RegistrationInterface(ABC):
    """
    Abstract base class for registration models.
    """

    @abstractmethod
    def __init__(self, configuration: RegistrationConfiguration):
        """
        Initialize the registration model.
        """

    @abstractmethod
    def register(self, fixed_image, moving_image):
        """
        Register moving_image to fixed_image.
        """

    @abstractmethod
    def get_transformation(self):
        """
        Return the transformation model.
        """
