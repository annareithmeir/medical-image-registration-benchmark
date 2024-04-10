from pathlib import Path
import subprocess
import os
import datetime

from typing import List

from ..core import TransformationInterface
from ..core.configurations import TransformationAffineNiftyRegConfiguration

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
        # example
        "reg_transform -ref referenceImage.nii -flo tumor.nii -trans transformation.txt -res outputImage.nii"
        date_time = datetime.datetime.now()

        output_path = fixed_image_path.parent / f"{moving_image_path.stem}_warped_on_{fixed_image_path.stem}_{self.method}_{date_time}.nii"
        output_path = Path(output_path.as_posix().replace(" ", "_"))

        command = [REG_TRANSFORM_PATH.as_posix()]
        command.extend(["-ref", fixed_image_path.as_posix()])
        command.extend(["-flo", moving_image_path.as_posix()])
        command.extend(["-trans", transformation_path.as_posix()])
        command.extend(["-res", output_path.as_posix()])

        self._print_command_line(command)

        try:
            p = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = p.communicate()

            if not output_path.exists():

                error_message = 'Outputs not written on the disk\n\n'
                error_message += str(output[1])

                raise FileNotFoundError(error_message)

        except OSError as e:
            print(e)
            print('Is reg_transform correctly installed?')
    
    @staticmethod
    def _print_command_line(command):
        print('\n\n')

        full_cmd = ""

        for a in command:
            full_cmd += a + " "

        print(full_cmd)
