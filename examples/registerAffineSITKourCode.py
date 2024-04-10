from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.affine_sitk import AffineSITK
from registrationbaselines.core.configurations import AffineSITKConfiguration, ResampleSITKConfiguration


def main():

    registration = AffineSITK(AffineSITKConfiguration(), ResampleSITKConfiguration())

    registration.register(Path("registrationbaselines/data/tumor1.nii"), Path("registrationbaselines/data/tumor2.nii"), True)

if __name__ == "__main__":
    main()
