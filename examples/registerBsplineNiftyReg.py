from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg
from registrationbaselines.data_loading import data_loaders


def main() -> None:

    # REGISTER A SINGLE PAIR OF IMAGES

    registration = BSplineNiftyReg(
        Path('registrationbaselines/configs/BSplineNiftyReg.yaml'))

    """
    registration.register(Path("/home/fryderyk/Documents/data/LungCT/imagesTr/LungCT_0001_0000.nii.gz"),
                          Path(
                              "/home/fryderyk/Documents/data/LungCT/imagesTr/LungCT_0001_0001.nii.gz"))
    """

    # REGISTER A BATCH OF IMAGES WITH A DATALOADER
    loader = data_loaders.L2RLungCTDataset(
        Path("/home/fryderyk/Documents/data/LungCT"))

    for fixed, moving in loader:
        registration.register(fixed, moving)


if __name__ == "__main__":
    main()
