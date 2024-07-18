from pathlib import Path
import warnings

from typing import List, Union, Any

import numpy as np
import SimpleITK as sitk

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline, utils_nifti


class DeformableCorrField(RegistrationInterface):
    """
    Deformable registration using corrField.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:

        self.method = "DeformableCorrField"

        self.configuration = self.read_config(configuration_path)

        self._create_result_directories()

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

            This is a 3 step process.
            1. Register both images to create correspondences - this needs a mask (correspondences are only searched in the mask area)
            2. Apply the correspondences to the moving image.
            3. The warped image has to be rotated by 180 degrees around the x-axis to be in the same orientation as the fixed image.
        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path

        self.__chek_all_inputs()

        self.__create_registration_command()
        utils_commandline.run_command_in_terminal(self.command_register,
                                                  self.correspondence_path.exists,
                                                  print_command_list=True)

        self.__create_transformation_command()
        utils_commandline.run_command_in_terminal(self.command_transform,
                                                  self.result_transformed_image_path.exists,
                                                  print_command_list=True)

        self.__rotate_warped_image_by_180_around_x_axis()

        # we don't need the mask anymore so let's delete it
        self.mask_path.unlink()

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        warnings.warn(
            "CorrField doesn't provide a transformation, only a correspondences file (for now).")

        return self.correspondence_path

    def _save_results(self, deformed, deformation):
        """
        Nothing happens here because saving is done thorugh the command line.
        """

    def set_fixed_mask(self, mask_path: Path) -> None:
        """
        You may want to set a non-dummy mask for the registration.
        """
        self.mask_path = mask_path

    def __chek_all_inputs(self):
        """
        Helper function to check all inputs.
        """
        # check that both images exist
        assert self.fixed_path.exists(
        ), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(
        ), f"File {self.moving_path} does not exist."

        # if no mask is given, create a dummy mask
        if self.mask_path == Path():
            self.mask_path = Path(
                self.fixed_path.as_posix().replace(".nii", "_mask.nii"))
            self.__create_empty_fixed_image_mask()
        assert self.mask_path.exists(
        ), f"File {self.mask_path} does not exist."

        # need to be ".nii.gz"
        assert self.fixed_path.suffixes == [
            ".nii", ".gz"], f"File {self.fixed_path} is not a .nii.gz file."
        assert self.moving_path.suffixes == [
            ".nii", ".gz"], f"File {self.moving_path} is not a .nii.gz file."
        assert self.mask_path.suffixes == [
            ".nii", ".gz"], f"File {self.moving_path} is not a .nii.gz file."

        # check all voxel sizes
        self.__check_if_image_has_isotropic_voxel_size(self.fixed_path)
        self.__check_if_image_has_isotropic_voxel_size(self.moving_path)
        self.__check_if_image_has_isotropic_voxel_size(self.mask_path)

    def __check_if_image_has_isotropic_voxel_size(self, image_path: Union[str, Path]) -> None:
        """
        Check if the input image has isotropic voxel size.

        @param image_path: Path to the input image file
        @type image_path: Union[str, Path]

        @return: None
        @rtype: None

        @raise ValueError: If the voxel size is not isotropic
        """
        # Read the image using SimpleITK
        image: sitk.Image = sitk.ReadImage(str(image_path))

        # Get the voxel spacing
        voxel_size: np.ndarray[Any, np.dtype[np.float64]
                               ] = np.array(image.GetSpacing())

        # Check if the voxel size is isotropic
        if not np.allclose(voxel_size, voxel_size[0]):
            raise ValueError(
                f"Voxel size of {image_path} is not isotropic: {tuple(voxel_size)}"
            )

    def __create_empty_fixed_image_mask(self) -> None:
        """
        Create an empty mask with the same dimensions as the fixed image and save it.

        This method reads the fixed image, creates a mask of ones with the same
        dimensions, and saves it to the specified mask path.

        @return: None
        @rtype: None
        """

        # Read the fixed image using SimpleITK
        fixed_image: sitk.Image = sitk.ReadImage(str(self.fixed_path))

        # Get the size of the fixed image
        size: tuple[int, ...] = fixed_image.GetSize()

        # Create a mask with the same dimensions as the fixed image
        mask: sitk.Image = sitk.Image(size, sitk.sitkUInt8)
        mask.CopyInformation(fixed_image)
        mask.FillBuffer(1)

        # Save the mask
        sitk.WriteImage(mask, str(self.mask_path))

    def __rotate_warped_image_by_180_around_x_axis(self) -> None:
        """
        Rotate the warped image by 180 degrees around the x-axis and save it.

        This method reads the transformed image, applies a 180-degree rotation
        around the x-axis, and saves the result back to the same file.

        @return: None
        @rtype: None
        """

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
        sitk.WriteImage(new_image, str(self.result_transformed_image_path))

    def __create_registration_command(self):
        """
        Create the command line list for the registration.
        """

        self.result_transformed_image_path, self.correspondence_path = self._create_result_paths(self.fixed_path.stem,
                                                                                                 self.moving_path.stem,
                                                                                                 ".nii.gz",
                                                                                                 ".dat")

        self.command_register = ["registrationbaselines/libraries/corrField_cpu/corrField_ubuntu",
                                 '-F', self.fixed_path.as_posix(),
                                 '-M', self.moving_path.as_posix(),
                                 '-m', self.mask_path.as_posix(),
                                 '-O', self.correspondence_path.as_posix()]

        self.command_register = utils_commandline.add_configuration_to_command(
            self.command_register, self.configuration)

    def __create_transformation_command(self):
        """
        Create the command line list for the registration.
        """

        self.command_transform = ["registrationbaselines/libraries/corrField_cpu/applyCorrField_ubuntu",
                                  '-M', self.moving_path.as_posix(),
                                  '-O', self.correspondence_path.as_posix(),
                                  '-W', self.result_transformed_image_path.as_posix()
                                  ]
