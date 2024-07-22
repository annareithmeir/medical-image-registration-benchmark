from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.syn_ants import SyNANTs
from registrationbaselines.data_loading import data_loaders


def main() -> None:

    registration = SyNANTs(Path('registrationbaselines/configs/SyNANTs.yaml'))

    loader = data_loaders.L2RLungCTDataset(dataset_path = "/home/anna/datasets/LungCT_preprocessed", return_type="path_dict")

    for item in loader:
        fixed = item["fixed_image"]
        moving = item["moving_image"]
        registration.register(fixed, moving)
        assert 3==4


if __name__ == "__main__":
    main()
