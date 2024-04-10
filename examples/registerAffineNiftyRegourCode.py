from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.affine_niftyreg import AffineNiftyReg
from registrationbaselines.core.configurations import AffineNiftyRegConfiguration


def main() -> None:

    registration = AffineNiftyReg(AffineNiftyRegConfiguration())

    registration.register(Path("registrationbaselines/data/tumor1.nii"), Path("registrationbaselines/data/tumor2.nii"))


if __name__ == "__main__":
    main()
