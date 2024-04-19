from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.affine_niftyreg import AffineNiftyReg


def main() -> None:

    registration = AffineNiftyReg(Path('registrationbaselines/configs/AffineNiftyReg.yaml'))

    registration.register(Path("registrationbaselines/data/unregistered/tumor1.nii"), Path("registrationbaselines/data/unregistered/tumor2.nii"))


if __name__ == "__main__":
    main()
