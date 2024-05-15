from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders


def main() -> None:

    evaluation = Evaluation(
        Path('registrationbaselines/configs/BSplineNiftyReg.yaml'))

    loader_data = data_loaders.L2RLungCTDataset(
        Path("/home/fryderyk/Documents/data/LungCT_small"))
    loader_transformations = data_loaders.BaselineTransformations(
        Path("/home/fryderyk/Documents/data/results/BSplineNiftyReg"))

    evaluation.evaluate(loader_transformations, loader_data)


if __name__ == "__main__":
    main()
