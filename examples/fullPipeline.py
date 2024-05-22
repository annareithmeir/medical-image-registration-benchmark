from pathlib import Path
import sys
import logging
import matplotlib

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.registration.syn_ants import SyNANTs
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders


def main() -> None:
    path_config = Path('registrationbaselines/configs/SyNANTs.yaml')
    path_data_root = Path("/home/fryderyk/Documents/data/LungCT_small")

    loader_data = data_loaders.L2RLungCTDataset(path_data_root)

    # register
    registration = SyNANTs(path_config)

    for i in range(5):
        registration.register(loader_data[i][0], loader_data[i][1])

    # evaluate
    evaluation = Evaluation(path_config)
    loader_transformations = data_loaders.BaselineTransformations(
        Path("/home/fryderyk/Documents/data/results/SyNANTs"))
    evaluation.evaluate(loader_transformations, loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    main()
