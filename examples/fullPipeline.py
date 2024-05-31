from pathlib import Path
import sys
import logging
import socket

import matplotlib
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg
from registrationbaselines.registration.convexadam import ConvexAdam
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.parent.absolute()

    # method = "BSplines"
    method = "convexAdam"

    if method == "BSplines":
        path_config = base_dir / 'registrationbaselines/configs/BSplineNiftyReg.yaml'
        registration = BSplineNiftyReg(path_config)
    elif method == "convexAdam":
        path_config = base_dir / 'registrationbaselines/configs/ConvexAdam.yaml'
        registration = ConvexAdam(path_config)
    else:
        print("Method not implemented")


    machine_name = socket.gethostname()
    if machine_name == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/LungCT")
    elif machine_name == "janus":
        path_data = Path("/u/home/koeglf/Documents/data/LungCT")
    else:
        path_data = Path("/home/anna/datasets/AbdomenMRCT_preprocessed")
        # path_data = Path("/home/anna/datasets/LungCT")

    indices = [0]
    loader_data = data_loaders.L2RAbdominalMRCTDataset(path_data, indices = indices)
    # loader_data = data_loaders.L2RLungCTDataset(path_data, indices = indices)

    # register
    print("\nregister...")
    for i in tqdm(range(len(loader_data))):
        item = loader_data[i]
        registration.register(item["images"][0], item["images"][1])

    if method == "BSplines":
        loader_transformations = data_loaders.BaselineTransformations(
            base_dir / "tmp/AbdomenMRCT/BSplineNiftyReg")
    elif method == "convexAdam":
        loader_transformations = data_loaders.BaselineTransformations(
            base_dir / "tmp/AbdomenMRCT/ConvexAdam")
    else:
        print("Method not implemented")

    # evaluate
    print("\nevaluate...")
    evaluation = Evaluation(path_config)
    evaluation.evaluate(loader_transformations, loader_data)
    print("\nplot...")
    evaluation.visualize(loader_transformations, loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    main()
