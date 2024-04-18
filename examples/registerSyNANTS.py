from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.syn_ants import SyNANTs


def main() -> None:

    registration = SyNANTs(Path('registrationbaselines/configs/SyNANTs.yaml'))

    registration.register(Path("registrationbaselines/data/tumor1.nii"), Path("registrationbaselines/data/tumor2.nii"), True)


if __name__ == "__main__":
    main()
