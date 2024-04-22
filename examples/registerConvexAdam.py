from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.convexadam import ConvexAdam


def main() -> None:
    base_dir = Path(__file__).parent.parent.absolute()

    registration = ConvexAdam(base_dir / 'registrationbaselines/configs/ConvexAdam.yaml')

    registration.register(base_dir / "registrationbaselines/data/tumor1.nii", base_dir / "registrationbaselines/data/tumor2.nii", True)


if __name__ == "__main__":
    main()