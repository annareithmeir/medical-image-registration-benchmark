from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.registration.affine_niftyreg import AffineNiftyReg


def main() -> None:

    registration = AffineNiftyReg(
        Path('registrationbaselines/configs/AffineNiftyReg.yaml'))

    registration.register(Path("/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/unregistered/tumor1.nii"),
                          Path("/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/unregistered/tumor2.nii"))


if __name__ == "__main__":
    main()
