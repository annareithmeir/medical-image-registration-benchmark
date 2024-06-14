from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.voxelmorph_feature import VoxelmorphFeature


def main() -> None:

    registration = VoxelmorphFeature(Path('registrationbaselines/configs/Voxelmorph_feature.yaml'))
    
    path_fixed = Path("/u/home/koeglf/Documents/data/LungCT/imagesTr/LungCT_0001_0000.nii.gz")
    path_moving = Path("/u/home/koeglf/Documents/data/LungCT/imagesTr/LungCT_0001_0001.nii.gz")

    registration.register(path_fixed, path_moving, True)


if __name__ == "__main__":
    main()
