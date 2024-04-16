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
        We first need to create a displacement field (set the intent code to NIFTI_INTENT_DISPVECT) from the affine transform and then apply it to the moving image.
        Then we apply the displacement field to the moving image.
        """

        path_displacement = utils_niftyreg.convert_affine_to_displacement_field(fixed_image_path, transformation_path)

        utils_niftyreg.apply_displacement_field(fixed_image_path, moving_image_path, path_displacement)

        os.remove(path_displacement)
    
    @staticmethod
    def _print_command_line(command):
        print('\n\n')

        full_cmd = ""

        for a in command:
            full_cmd += a + " "

        print(full_cmd)
