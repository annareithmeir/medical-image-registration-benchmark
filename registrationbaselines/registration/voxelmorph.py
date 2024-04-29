from pathlib import Path
import sys

from registrationbaselines.core import utils_commandline
from registrationbaselines.registration._interface_registration import RegistrationInterface


class VoxelmorphReg(RegistrationInterface):
    def __init__(self, configuration_path: Path):
        pass
    
    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        pass
    
    def get_transformation_path(self):
        pass
    
    def get_transformed_image_path(self):
        pass