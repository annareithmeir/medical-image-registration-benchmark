from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class TransformationInterface(ABC):
    """
    Abstract base class for transformations.
    """
    
    configuration = None
    
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

    @staticmethod
    def read_config(file_path: Path):
        """
        Read the configuration file.
        """
        
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        