from pathlib import Path

from registrationbaselines.core import utils_niftyreg
from registrationbaselines.transforms._interface_transformation import TransformationInterface
from registrationbaselines.core.configurations import TransformationBSplineNiftyRegConfiguration

REG_TRANSFORM_PATH = Path('/usr/local/bin/reg_transform')


class TransformBSplineNiftyReg(TransformationInterface):
    """
    BSpline transformation using NiftyReg.
    """

    def __init__(self, configuration: TransformationBSplineNiftyRegConfiguration):
        """
        Initialize the transformation model.
        """
        self.transformation_type = configuration.transformation_type
        self.remaining_arguments = configuration.remaining_arguments

        self.method = "BSplineNiftyReg"
    
    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        return utils_niftyreg.apply_transformation(fixed_image_path, moving_image_path, transformation_path)