from pathlib import Path
import sys
import logging
import socket
import os
import time
import SimpleITK as sitk

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8

from registrationbaselines.core import utils  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.evaluation.evaluation import Evaluation  # nopep8
from registrationbaselines.training.train_voxelmorph_feature import VoxelmorphFeatureTraining  # nopep8
from registrationbaselines.registration.syn_ants import SyNANTs  # nopep8
from registrationbaselines.registration.voxelmorph import VoxelmorphReg  # nopep8
from registrationbaselines.registration.bspline_feature import BSplineFeature  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.parent.absolute()

    method = "SyNANTs"
    # method = "BSplines"
    # method = "voxelmorph_feature"

    # path_config = base_dir / f"registrationbaselines/configs/BSplines_feat.yaml"
    path_config = base_dir / f"registrationbaselines/configs/{method}.yaml"
    config = utils.read_config(path_config)

    if method == "BSplines":
        registration = BSplineFeature(config)
    elif method == "SyNANTs":
        registration = SyNANTs(path_config)
    else:
        raise ValueError("Method not implemented")

    machine_name = socket.gethostname()
    if machine_name == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/ACDC/")
    elif machine_name == "janus":
        path_data = Path("/data/ACDC/database/")
    else:
        path_data = Path("/home/anna/datasets/LungCT_preprocessed")

        # path_data = Path("/home/anna/datasets/FIRE")

    loader_data = data_loaders.L2RLungCTDataset(
        path_data, return_type="path_dict")
    # loader_data.preprocess(Path("/home/anna/datasets/LungCT_preprocessed"))

    for item in loader_data:
        fixed = item["fixed_image"]
        moving = item["moving_image"]
        # registration.register(fixed, moving)
        break

    # registration.register_all_parametr_sets(loader_data)

    print("\nevaluate...")
    loader_transformations = data_loaders.BaselineTransformations(
        Path(config["parameters"]["result_path"]["value"]) / method)

    evaluation = Evaluation(
        Path(config["parameters"]["result_path"]["value"]), method)
    evaluation.evaluate(
        loader_transformations, loader_data)
    print("\nplot...")
    evaluation.visualize(
        loader_transformations, loader_data)
    print("\ndone\n\n")

    evaluation.evaluate(loader_transformations, loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    # matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    start_time = time.time()
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Save execution time to a tex file
    with open('/home/anna/PycharmProjects/registrationbaselines/tmp/time.txt', 'w') as f:
        # with open('/u/home/koeglf/Documents/code/registrationbaselines/tmp/time.txt', 'w') as f:
        f.write(f"Execution time: {execution_time} seconds")
