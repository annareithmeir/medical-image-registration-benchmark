from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.transforms.transform_syn_ants import TransformSyNANTs
from registrationbaselines.registration.syn_ants import SyNANTs 


def main():
    """
    Here we first register and then use the registration to deform the image.
    The registration already outputs the transformed file, but we can check if our transformation works.
    """

    path_fixed = Path("registrationbaselines/data/tumor1.nii")
    path_moving = Path("registrationbaselines/data/tumor2.nii")

    registration = SyNANTs(Path('registrationbaselines/configs/BSplineNiftyReg.yaml'))
    registration.register(path_fixed, path_moving)

    transformation = TransformSyNANTs()
    transformation.apply_transformation(path_fixed,
                                        path_moving,
                                        registration.get_transformation_path())


if __name__ == "__main__":
    main()
