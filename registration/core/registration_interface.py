from abc import ABC, abstractmethod


class RegistrationInterface(ABC):
    @abstractmethod
    def register(self, fixed_image, moving_image):
        """
        Register moving_image to fixed_image.
        """
        pass

    @abstractmethod
    def get_transformation(self):
        """
        Return the transformation model.
        """
        pass
