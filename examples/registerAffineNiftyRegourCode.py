from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from registration.core.configurations import AffineNiftyRegConfiguration
from registration.integrations.affine_niftyreg import AffineNiftyReg


def main():

    registration = AffineNiftyReg(AffineNiftyRegConfiguration())

    registration.register(Path("data/tumor1.nii"), Path("data/tumor2.nii"))

if __name__ == "__main__":
    main()
