from pathlib import Path
import subprocess
import os
import datetime

from typing import List

from registrationbaselines.core import utils_niftyreg
from registrationbaselines.transforms._interface_transformation import TransformationInterface
from registrationbaselines.core.configurations import TransformationAffineNiftyRegConfiguration

REG_TRANSFORM_PATH = Path('/usr/local/bin/reg_transform')


class TransformAffineNiftyReg(TransformationInterface):
    """
    Affine transformation using NiftyReg.
    """

    def __init__(self, configuration: TransformationAffineNiftyRegConfiguration):
        """
        Initialize the transformation model.
        """
        self.transformation_type = configuration.transformation_type
        self.remaining_arguments = configuration.remaining_arguments

        self.method = "AffineNiftyReg"
    
    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        return utils_niftyreg.apply_transformation(fixed_image_path, moving_image_path, transformation_path)
    
    @staticmethod
    def _print_command_line(command):
        print('\n\n')

        full_cmd = ""

        for a in command:
            full_cmd += a + " "

        print(full_cmd)
