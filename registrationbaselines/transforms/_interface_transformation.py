from abc import ABC, abstractmethod
from pathlib import Path


class TransformationInterface(ABC):
    """
        Abstract base class for transformations.
    """

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
