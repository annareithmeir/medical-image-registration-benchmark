from pathlib import Path
import sys
import logging
import matplotlib

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders


def main() -> None:
    base_dir = Path(__file__).parent.parent.absolute()

    evaluation = Evaluation(base_dir / 'registrationbaselines/configs/BSplineNiftyReg.yaml')

    loader_data = data_loaders.L2RLungCTDataset(Path("/home/anna/datasets/LungCT"), idxs=[0])
    loader_transformations = data_loaders.BaselineTransformations( Path("/home/anna/PycharmProjects/registrationbaselines/tmp/results"))

    evaluation.evaluate(loader_transformations, loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    main()
