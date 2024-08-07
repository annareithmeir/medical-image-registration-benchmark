from pathlib import Path
import sys
import logging
import socket
import os
import time

from typing import Any, Dict

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
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8


def get_non_wand_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the config without the wandb config.
    """

    config = config["parameters"]

    new_config: Dict[str, str] = {}

    for key, value in config.items():
        new_config[key] = value["values"][0]

    return new_config


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.parent.absolute()

    method = "SyNANTs"
    # method = "BSplines"
    # method = "voxelmorph_feature"

    path_config = base_dir / f"registrationbaselines/configs/{method}.yaml"
    config = utils.read_config(path_config)

    config = get_non_wand_config(config)

    machine_name = socket.gethostname()
    if machine_name == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/LungCT_preprocessed/")
    elif machine_name == "janus":
        path_data = Path("/data/LungCT_preprocessed")
    else:
        path_data = Path("/home/anna/datasets/LungCT_preprocessed")
        # path_data = Path("/home/anna/datasets/FIRE")

    loader_data = data_loaders.L2RLungCTDataset(path_data,
                                                return_type="path_dict",
                                                indices=[0])
    # loader_data.preprocess(Path("/home/anna/datasets/LungCT_preprocessed"))
    use_wandb = False

    if method == "BSplines":
        registration = BSplineNiftyReg(config,
                                       loader_data,
                                       use_wandb=False)
    elif method == "SyNANTs":
        registration = SyNANTs(config, loader_data)
    else:
        raise ValueError("Method not implemented")


    registration._create_result_directories(method)

    for item in loader_data:
        fixed = item["fixed_image"]
        moving = item["moving_image"]
        registration.register(fixed, moving)

    # registration.register_all_parametr_sets(loader_data)

    print("\nevaluate...")

    result_path = Path(config["parameters"]["result_path"]
                       ["value"]) if use_wandb else Path(config["result_path"])

    loader_transformations = data_loaders.BaselineTransformations(
        registration.method_dir)

    evaluation = Evaluation(result_path,
                            method,
                            loader_data,
                            loader_transformations)
    # evaluation.evaluate()
    print("\nplot...")

    evaluation.visualize()
    print("\ndone\n\n")


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
