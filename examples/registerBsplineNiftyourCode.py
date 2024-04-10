from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg
from registrationbaselines.core.configurations import BSplineNiftyRegConfiguration


def main():

    registration = BSplineNiftyReg(BSplineNiftyRegConfiguration())

    registration.register(Path("registrationbaselines/data/tumor1.nii"), Path("registrationbaselines/data/tumor2.nii"), True)

if __name__ == "__main__":
    main()
