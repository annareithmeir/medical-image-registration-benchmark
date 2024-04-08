from abc import ABC, abstractmethod


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
    def register(self, fixed_image, moving_image, print_progress=False):
        """
        Register moving_image to fixed_image.
        """

    @abstractmethod
    def get_transformation(self):
        """
        Return the transformation model.
        """
