from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.convexadam import ConvexAdam
from registrationbaselines.data_loading import data_loaders


def main() -> None:
    base_dir = Path(__file__).parent.parent.absolute()

    registration = ConvexAdam(base_dir / 'registrationbaselines/configs/ConvexAdam.yaml')

    loader = data_loaders.L2RLungCTDataset(
        Path("/home/anna/datasets/LungCT"))

    for fixed, moving in loader:
        registration.register(fixed, moving)

if __name__ == "__main__":
    main()