from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_deformable_corrfield import TransformDeformableCorrField
from registrationbaselines.registration.deformable_corrfield import DeformableCorrField


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz").absolute()
    path_moving = Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2_resampled111_normalized.nii.gz").absolute()

    registration = DeformableCorrField(Path('registrationbaselines/configs/DeformableCorrField.yaml'))
    registration.register(path_fixed, path_moving)

    transformation = TransformDeformableCorrField()
    transformation.apply_transformation(path_fixed,
                                        path_moving,
                                        registration.get_transformation_path())


if __name__ == "__main__":
    main()
