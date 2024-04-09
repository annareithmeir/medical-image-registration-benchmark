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

    registration = AffineSITK(AffineSITKConfiguration(), ResampleSITKConfiguration())

    registration.register(fixed, moving, True)

    sitk.WriteTransform(registration.get_transformation(), r"transform.hdf5")
    sitk.WriteImage(registration.get_transformed_image(), r"output.nii")

if __name__ == "__main__":
    main()