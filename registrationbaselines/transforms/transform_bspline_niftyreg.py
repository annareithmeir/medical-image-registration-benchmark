from pathlib import Path

from registrationbaselines.core import utils_niftyreg
from registrationbaselines.transforms._interface_transformation import TransformationInterface


class TransformBSplineNiftyReg(TransformationInterface):
    """
    BSpline transformation using NiftyReg.
    """

    def __init__(self, configuration_path: Path) -> None:
        """
        Initialize the transformation model.
        """

        self.method = "BSplineNiftyReg"

        self.configuration = self.read_config(configuration_path)

        self._create_result_directories()

    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        path_deformed = self._create_result_path(fixed_image_path.stem,
                                                 moving_image_path.stem,
                                                 ".nii.gz")

        return utils_niftyreg.apply_transformation(fixed_image_path, moving_image_path, transformation_path, path_deformed)

    def _save_results(self, deformed):
        """
        Nothing happens here because saving is done thorugh the command line.
        """
