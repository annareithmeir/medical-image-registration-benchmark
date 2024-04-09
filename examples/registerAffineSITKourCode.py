from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from registration.core.configurations import AffineSITKConfiguration, ResampleSITKConfiguration
from registration.integrations.affine_sitk import AffineSITK


def main():

    registration = AffineSITK(AffineSITKConfiguration(), ResampleSITKConfiguration())

    registration.register(Path("data/tumor1.nii"), Path("data/tumor2.nii"), True)

if __name__ == "__main__":
    main()