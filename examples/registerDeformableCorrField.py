from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.deformable_corrfield import DeformableCorrField


def main() -> None:

    registration = DeformableCorrField(Path('registrationbaselines/configs/DeformableCorrField.yaml'))

    registration.register(Path("registrationbaselines/data/tumor1.nii.gz"), Path("registrationbaselines/data/tumor2.nii.gz"), True)


if __name__ == "__main__":
    main()
