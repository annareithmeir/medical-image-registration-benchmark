from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_demons_sitk import TransformDemonsSITK
from registrationbaselines.registration.demons_sitk import DemonsSITK


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz")
    path_moving = Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2_resampled111_normalized.nii.gz")

    registration = DemonsSITK(Path('registrationbaselines/configs/DemonsSITK.yaml'),
                              Path('registrationbaselines/configs/ResampleDemonsSITK.yaml'))
    registration.register(path_fixed, path_moving)

    transformation = TransformDemonsSITK(Path('registrationbaselines/configs/ResampleDemonsSITK.yaml'))
    path_transformed = transformation.apply_transformation(path_fixed, path_moving, registration.get_transformation_path())

    print(f"Transformed image: {path_transformed}")

if __name__ == "__main__":
    main()
