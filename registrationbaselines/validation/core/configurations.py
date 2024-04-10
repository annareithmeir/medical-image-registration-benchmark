from typing import List

from registration.core.enums import TransformationType

class TransformationAffineNiftyRegConfiguration():
    """
    Configuration class for the NiftyReg affine transformation.
    """

    def __init__(self, transformation_type: TransformationType = TransformationType.AFFINE,
                 remaining_arguments: List[str] = None) -> None:
        """
        Args:
            transformation_type (TransformationType): Type of transformation to use.
            remaining_arguments (List[str]): Remaining arguments to pass to NiftyReg. has to be a list of strings, like so:
            ["-noSym", "-ln", "5"]
        """

        self.transformation_type = transformation_type
        self.remaining_arguments = remaining_arguments