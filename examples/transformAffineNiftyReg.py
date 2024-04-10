from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from validation.integrations.transform_affine_niftyreg import TransformAffineNiftyReg
from validation.core.configurations import TransformationAffineNiftyRegConfiguration


def main():

    transformation = TransformAffineNiftyReg(TransformationAffineNiftyRegConfiguration())

    transformation.apply_transformation(Path("data/tumor1.nii"), Path("data/tumor2.nii"), Path("data/tumor2_warped_on_tumor1_AffineNiftyReg_transform_2024-04-09_16:54:20.832791.txt"))

if __name__ == "__main__":
    main()
