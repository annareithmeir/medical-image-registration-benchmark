from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg


def main() -> None:

    registration = BSplineNiftyReg(Path('registrationbaselines/configs/BSplineNiftyReg.yaml'))

    registration.register(Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1.nii.gz"),
                          Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2.nii.gz"), True)


if __name__ == "__main__":
    main()
