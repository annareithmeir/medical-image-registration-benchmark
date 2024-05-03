import ants
import os
from pathlib import Path

from registrationbaselines.core import utils_commandline
from registrationbaselines.registration._interface_registration import RegistrationInterface


class SyNANTs(RegistrationInterface):
    """
    SyN registration using ANTs.
    No default initialisation, as the choice of registration should be concious.
    """
    
    def __init__(self, configuration_path: Path) -> None:
        """
        Initialize the registration model.
        """
        
        self.method = "SyNANTs"
        
        self.configuration = self.read_config(configuration_path)
        
        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_transformation_path = Path()
    
    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        """
        Wrapper around ants to register.
        """
        
        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path
        
        # check that both images exist
        assert self.fixed_path.exists(), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(), f"File {self.moving_path} does not exist."
        
        # load boath images with ants
        fixed_image = ants.image_read(self.fixed_path.as_posix())
        moving_image = ants.image_read(self.moving_path.as_posix())
        
        # Perform registration
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='SyNOnly',
            write_composite_transform=True  # nopep8 this outputs one .h5 transform, otherwise we have a .nii.gz and .mat
        )
        
        self.__save_results(registration)

    def get_transformation_path(self):
        return self.result_transformation_path
    
    def get_transformed_image_path(self):
        return self.result_transformed_image_path
    
    def __save_results(self, registration):
        self.result_transformed_image_path, self.result_transformation_path = \
            utils_commandline.create_result_paths(self.fixed_path.parent,
                                                  self.fixed_path.stem,
                                                  self.moving_path.stem,
                                                  self.method,
                                                  ".nii",
                                                  ".h5")
        
        moving_image_warped = registration['warpedmovout']
        moving_image_warped.to_filename(self.result_transformed_image_path)
        
        # the transformation is already save in a temp folder, so we just move it
        os.rename(registration['fwdtransforms'], self.result_transformation_path)
