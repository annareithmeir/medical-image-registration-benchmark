from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_bspline_niftyreg import TransformBSplineNiftyReg
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1.nii.gz")
    path_moving = Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2.nii.gz")

    registration = BSplineNiftyReg(Path('registrationbaselines/configs/BSplineNiftyReg.yaml'))
    registration.register(path_fixed, path_moving)

    transformation = TransformBSplineNiftyReg()
    transformation.apply_transformation(path_fixed, path_moving, registration.get_transformation_path())


if __name__ == "__main__":
    main()
