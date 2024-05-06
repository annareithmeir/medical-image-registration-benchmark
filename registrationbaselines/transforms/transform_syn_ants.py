from pathlib import Path

import ants

from registrationbaselines.transforms._interface_transformation import TransformationInterface


class TransformSyNANTs(TransformationInterface):
    """
    SyN transformation using ANTs.
    """

    def __init__(self, configuration_path: Path) -> None:
        """
        Initialize the transformation model.
        """

        self.method = "SyNANTs"

        self.configuration = self.read_config(configuration_path)

        self._create_result_directories()

        self.path_fixed = Path()
        self.path_moving = Path()

    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        moving_image_warped = ants.apply_transforms(fixed=ants.image_read(self.path_fixed.as_posix()),
                                                    moving=ants.image_read(self.path_moving.as_posix()),  # nopep8
                                                    transformlist=[transformation_path.as_posix()])

        path_output = self._save_results(moving_image_warped)

        return path_output

    def _save_results(self, deformed):
        path_output = self._create_result_path(self.path_fixed.stem,
                                               self.path_moving.stem,
                                               ".nii.gz")

        deformed.to_filename(path_output)

        if not path_output.exists():
            raise ValueError(f"Error saving the file: {path_output}")

        return path_output
