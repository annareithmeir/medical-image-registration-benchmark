from enum import Enum
import numpy as np


class TransformationType(Enum):
    AFFINE = 1
    NON_RIGID = 2


class Transformation:

    def __init__(self, transformation_type: TransformationType):
        self.transformation_type = transformation_type

    def apply_transformation(self, image, transformation):
        """
        Apply a transformation to an image using the provided matrix.
        """
        if self.transformation_type == TransformationType.AFFINE:
            self._apply_affine_transformation(image, transformation)
        elif self.transformation_type == TransformationType.NON_RIGID:
            self._apply_non_rigid_transformation(image, transformation)
        else:
            raise ValueError("Invalid transformation type")

    def _apply_affine_transformation(self, image, transformation_matrix):
        """
        Apply an affine transformation to an image using the provided matrix.
        """
        pass

    def _apply_non_rigid_transformation(self, image, transformation_field):
        """
        Apply a non-rigid transformation to an image based on provided field..
        """
        pass
