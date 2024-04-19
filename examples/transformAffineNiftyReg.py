from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_affine_niftyreg import TransformAffineNiftyReg
from registrationbaselines.registration.affine_niftyreg import AffineNiftyReg


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/unregistered/tumor1.nii")
    path_moving = Path("registrationbaselines/data/unregistered/tumor2.nii")

    registration = AffineNiftyReg(Path('registrationbaselines/configs/AffineNiftyReg.yaml'))
    registration.register(path_fixed, path_moving)

    transformation = TransformAffineNiftyReg()
    path_transformed = transformation.apply_transformation(path_fixed, path_moving, registration.get_transformation_path())

    print(f"Transformed image: {path_transformed}")


if __name__ == "__main__":
    main()
