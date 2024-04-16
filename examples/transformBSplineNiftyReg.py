from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_bspline_niftyreg import TransformBSplineNiftyReg
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg
from registrationbaselines.core.configurations import TransformationBSplineNiftyRegConfiguration, BSplineNiftyRegConfiguration


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/tumor1.nii")
    path_moving = Path("registrationbaselines/data/tumor2.nii")

    registration = BSplineNiftyReg(BSplineNiftyRegConfiguration())
    registration.register(path_fixed, path_moving)

    transformation = TransformBSplineNiftyReg(TransformationBSplineNiftyRegConfiguration())
    path_transformed = transformation.apply_transformation(path_fixed, path_moving, registration.get_transformation_path())

    print(f"Transformed image: {path_transformed}")


if __name__ == "__main__":
    main()
