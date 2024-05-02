from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.syn_ants import SyNANTs


def main() -> None:

    registration = SyNANTs(Path('registrationbaselines/configs/SyNANTs.yaml'))

    registration.register(Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor1.nii.gz"),
                          Path("registrationbaselines/data/affinely_registered_NiftyReg/tumor2.nii.gz"), True)


if __name__ == "__main__":
    main()
