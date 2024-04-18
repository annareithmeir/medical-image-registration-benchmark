from pathlib import Path
import os
import warnings

from typing import List

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline


class DeformableCorrField(RegistrationInterface):
    """
    Deformable registration using corrField.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:

        self.method = "DeformableCorrField"

        self.configuration = self.read_config(configuration_path)

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.correspondence_path = Path()
        self.mask_path = Path()

        # command to call NiftyReg
        self.command_register: List[str] = []
        self.command_transform: List[str] = []

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False) -> None:
        """
            Registration using corrField.
        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path
        self.mask_path = self.fixed_path  # todo we need a real mask

        # check that both images exist
        assert self.fixed_path.exists(), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(), f"File {self.moving_path} does not exist."
        
        # need to be ".nii.gz"
        assert self.fixed_path.suffixes == [".nii", ".gz"], f"File {self.fixed_path} is not a .nii.gz file."
        assert self.moving_path.suffixes == [".nii", ".gz"], f"File {self.moving_path} is not a .nii.gz file."
        
        self._create_registration_command()
        utils_commandline.print_command(self.command_register)
        utils_commandline.run_command_in_terminal(self.command_register, self.correspondence_path.exists)

        self._create_transformation_command()
        utils_commandline.print_command(self.command_transform)
        utils_commandline.run_command_in_terminal(self.command_transform, self.result_transformed_image_path.exists)
        
    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation
        
        warnings.warn("CorrField doesn't provide a transformation, only a correspondences file.")
        

        return self.correspondence_path

    def set_fixed_mask(self, mask_path: Path) -> None:
        """
        You may want to set a non-dummy mask for the registration.
        """
        self.mask_path = mask_path
    
    def __check_if_image_has_isotropic_voxel_size(self, image_path: Path) -> None:
        # Check if the voxel size is isotropic
        image = nib.load(image_path)
        voxel_size = image.header.get_zooms()
        
        if not np.allclose(voxel_size, voxel_size[0]):
            raise ValueError(f"Voxel size of {image_path} is not isotropic: {voxel_size}")
    
    def __create_empty_fixed_image_mask(self) -> None:
        # Create a mask with the same dimensions as the fixed image
        fixed_image = nib.load(self.fixed_path)
        mask = np.ones(fixed_image.shape)
        
        # Save the mask
        nib.save(nib.Nifti1Image(mask, fixed_image.affine), self.mask_path)
    
    def __rotate_warped_image_by_180_around_x_axis(self) -> None:
        # transform the image
        rotation_matrix_180_around_x = np.array([
            [1, 0,  0, 0],
            [0, -1, 0, 0],  # Invert y-axis
            [0, 0, -1, 0],  # Invert z-axis
            [0, 0,  0, 1]
        ])
        new_image = utils_nifti.transform_nifti_image_with_matrix(self.result_transformed_image_path,
                                                                  rotation_matrix_180_around_x,
                                                                  just_replace_existing_affine=False)

        # Save the transformed image
        nib.save(new_image, self.result_transformed_image_path)
        
    def __create_registration_command(self):
        """
        Create the command line list for the registration.
        """

        self.result_transformed_image_path, self.correspondence_path = utils_commandline.create_result_paths(self.fixed_path.parent,
                                                                                                             self.fixed_path.stem,
                                                                                                             self.moving_path.stem,
                                                                                                             self.method,
                                                                                                             ".nii.gz",
                                                                                                             ".dat")

        self.command_register = ["registrationbaselines/libraries/corrField_cpu/corrField_ubuntu",
                        '-F', self.fixed_path.as_posix(),
                        '-M', self.moving_path.as_posix(),
                        '-m', self.fixed_path.as_posix(),  
                        '-O', self.correspondence_path.as_posix()]
        
        self.command_register = utils_commandline.add_configuration_to_command(self.command_register, self.configuration)

    def __create_transformation_command(self):
        """
        Create the command line list for the registration.
        """

        self.command_transform = ["registrationbaselines/libraries/corrField_cpu/applyCorrField_ubuntu",
                                  '-M', self.moving_path.as_posix(),
                                  '-O', self.correspondence_path.as_posix(),
                                  '-W', self.result_transformed_image_path.as_posix()
                                  ]
    
    