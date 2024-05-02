from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.demons_sitk import DemonsSITK


def main() -> None:

    registration = DemonsSITK(Path('registrationbaselines/configs/DemonsSITK.yaml'), Path('registrationbaselines/configs/ResampleDemonsSITK.yaml'))

    registration.register(Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz"),
                          Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2_resampled111_normalized.nii.gz"))


if __name__ == "__main__":
    main()
