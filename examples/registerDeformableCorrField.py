from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.deformable_corrfield import DeformableCorrField


def main() -> None:

    registration = DeformableCorrField(Path('registrationbaselines/configs/DeformableCorrField.yaml'))

    registration.register(Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz").absolute(),
                          Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2_resampled111_normalized.nii.gz").absolute())


if __name__ == "__main__":
    main()
