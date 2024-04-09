from pathlib import Path
import sys

import SimpleITK as sitk

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from registration.core.configurations import AffineSITKConfiguration, ResampleSITKConfiguration
from registration.integrations.affine_sitk import AffineSITK


def main():

    fixed = sitk.ReadImage(r"data/tumor1.nii", sitk.sitkFloat32)
    moving = sitk.ReadImage(r"data/tumor2.nii", sitk.sitkFloat32)

    R = AffineSITK(AffineSITKConfiguration(), ResampleSITKConfiguration())

    R.register(fixed, moving, True)

    transformation = R.get_transformation()

    sitk.WriteTransform(transformation, r"transform.hdf5")

if __name__ == "__main__":
    main()