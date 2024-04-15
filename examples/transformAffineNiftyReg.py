from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_affine_niftyreg import TransformAffineNiftyReg
from registrationbaselines.registration.affine_niftyreg import AffineNiftyReg
from registrationbaselines.core.configurations import TransformationAffineNiftyRegConfiguration, AffineNiftyRegConfiguration


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/tumor1.nii")
    path_moving = Path("registrationbaselines/data/tumor2.nii")

    registration = AffineNiftyReg(AffineNiftyRegConfiguration())
    start_time = time.time()
    registration.register(path_fixed, path_moving)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    transformation = TransformAffineNiftyReg(TransformationAffineNiftyRegConfiguration())
    path_transformed = transformation.apply_transformation(path_fixed, path_moving, registration.transformation_path)

    print(f"Transformed image: {path_transformed}")


if __name__ == "__main__":
    main()
