from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders


def main() -> None:

    evaluation = Evaluation(
        Path('registrationbaselines/configs/BSplineNiftyReg.yaml'))

    loader_data = data_loaders.L2RLungCTDataset(
        Path("/home/fryderyk/Documents/data/LungCT"),
        return_segmentation=True)
    loader_transformations = data_loaders.BaselineTransformations(
        Path("/home/fryderyk/Documents/data/results/BSplineNiftyReg"))

    for i in range(min(len(loader_data), len(loader_transformations))):
        path_fixed_segmentation, path_moving_segmentation = loader_data[i]
        path_transformation = loader_transformations[i]

        evaluation.evaluate(path_transformation,
                            path_fixed_segmentation,
                            path_moving_segmentation)


if __name__ == "__main__":
    main()
