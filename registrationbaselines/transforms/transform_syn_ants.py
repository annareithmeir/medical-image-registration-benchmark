from pathlib import Path

import ants

from registrationbaselines.core import utils_commandline
from registrationbaselines.transforms._interface_transformation import TransformationInterface


class TransformSyNANTs(TransformationInterface):
    """
    SyN transformation using ANTs.
    """

    def __init__(self):
        """
        Initialize the transformation model.
        """
        
        self.method = "SyNANTs"
    
    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """
        
        moving_image_warped = ants.apply_transforms(fixed=ants.image_read(fixed_image_path.as_posix()),
                                                    moving=ants.image_read(moving_image_path.as_posix()),
                                                    transformlist=[transformation_path.as_posix()])
        
        path_output, _ = utils_commandline.create_result_paths(fixed_image_path.parent, fixed_image_path.stem, moving_image_path.stem, self.method, ".nii", ".nii")

        moving_image_warped.to_filename(path_output)
        
        if not path_output.exists():
            raise ValueError(f"Error saving the file: {path_output}")
        
        return path_output
