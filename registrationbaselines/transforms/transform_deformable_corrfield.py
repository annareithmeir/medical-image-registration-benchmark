from pathlib import Path

import numpy as np
import nibabel as nib

from registrationbaselines.transforms._interface_transformation import TransformationInterface
from registrationbaselines.core import utils_commandline, utils_nifti


class TransformDeformableCorrField(TransformationInterface):
    """
    Deformable transformation using corrField.
    """

    def __init__(self):
        """
        Initialize the transformation model.
        """

        self.method = "DeformableCorrField"

        self.fixed_path = Path()
        self.moving_path = Path()
        self.correspondence_path = Path()

        self.command = []

        self.result_transformed_image_path = Path()

    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path
        self.correspondence_path = transformation_path

        self.__create_transformation_command()
        utils_commandline.run_command_in_terminal(self.command,
                                                  self.result_transformed_image_path.exists,
                                                  print_command_list=True)

        self.__rotate_warped_image_by_180_around_x_axis()

        return self.result_transformed_image_path

    def __save_results(self, deformed):
        """
        Nothing happens here because saving is done thorugh the command line.
        """

    def __create_transformation_command(self):
        """
        Create the command line list for the registration.
        """

        self.result_transformed_image_path, _ = utils_commandline.create_result_paths(self.fixed_path.parent,
                                                                                      self.fixed_path.stem,
                                                                                      self.moving_path.stem,
                                                                                      self.method,
                                                                                      ".nii.gz",
                                                                                      ".dat")

        self.command = ["registrationbaselines/libraries/corrField_cpu/applyCorrField_ubuntu",
                        '-M', self.moving_path.as_posix(),
                        '-O', self.correspondence_path.as_posix(),
                        '-W', self.result_transformed_image_path.as_posix()
                        ]

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
