from pathlib import Path
import sys
import logging
import matplotlib
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders


def main() -> None:
    base_dir = Path(__file__).parent.parent.absolute()

    path_config = base_dir / 'registrationbaselines/configs/BSplineNiftyReg.yaml'

    loader_data = data_loaders.L2RLungCTDataset(Path("/home/anna/datasets/LungCT"), idxs=[0,1,2]) # we only want to use one image pair here

    # register
    print("\nregister...")
    registration = BSplineNiftyReg(path_config)

    for i in tqdm(range(len(loader_data))):
        item = loader_data[i]
        registration.register(item["images"][0], item["images"][1])

    # evaluate
    print("\nevaluate...")
    evaluation = Evaluation(path_config)
    loader_transformations = data_loaders.BaselineTransformations(
        base_dir / "tmp/results/BSplineNiftyReg")
    evaluation.evaluate(loader_transformations, loader_data)
    print("\nplot...")
    evaluation.visualize(loader_transformations, loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    main()
