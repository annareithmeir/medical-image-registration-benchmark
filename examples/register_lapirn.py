from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.lapirn import LapIRNReg


def main() -> None:
    base_dir = Path(__file__).parent.parent.absolute()
    registration = LapIRNReg(base_dir / Path('registrationbaselines/configs/TrainLapirn.yaml'))

    path_fixed = Path("/home/anna/datasets/LungCT/imagesTr/LungCT_0001_0000.nii.gz")
    path_moving = Path("/home/anna/datasets/LungCT/imagesTr/LungCT_0001_0001.nii.gz")

    registration.register( path_fixed, path_moving, True)


if __name__ == "__main__":
    main()
