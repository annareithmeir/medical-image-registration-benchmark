from pathlib import Path

from registrationbaselines.core import utils_niftyreg
from registrationbaselines.transforms._interface_transformation import TransformationInterface


class TransformBSplineNiftyReg(TransformationInterface):
    """
    BSpline transformation using NiftyReg.
    """

    def __init__(self):
        """
        Initialize the transformation model.
        """
    
    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        return utils_niftyreg.apply_transformation(fixed_image_path, moving_image_path, transformation_path)