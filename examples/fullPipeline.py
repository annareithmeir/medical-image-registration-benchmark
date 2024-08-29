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

    # method = "SyNANTs"
    method = "DemonsSITK"
    # method = "BSplineNiftyReg"
    # method = "AffineNiftyReg"
    # method = "VoxelMorph"

    config = get_non_wand_config(config)

    machine_name = socket.gethostname()
    if machine_name == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/LungCT_preprocessed/")
    elif machine_name == "janus":
        path_data = Path("/data/LungCT_preprocessed")
    else:
        path_data = Path("/home/anna/datasets/AbdomenMRCT_preprocessed")

    #####################################################################################################
    # REGISTER A REAL DATASET
    #####################################################################################################
    loader_data = data_loaders.L2RLungCTDataset(path_data,
                                                return_type="path_dict",
                                                indices=[i for i in range(2)])
    # loader_data = data_loaders.L2RAbdominalMRCTDataset(path_data,
    #                                                    return_type="path_dict",
    #                                                    indices=[0, 1])
    # loader_data = data_loaders.ImagePairDataset([[Path("/u/home/koeglf/Documents/code/registrationbaselines/fixed_x_11.nii.gz"),
    #                                               Path("/u/home/koeglf/Documents/code/registrationbaselines/moving_x_11.nii.gz")]],
    #                                             return_type="path_dict")

    if method == "BSplineNiftyReg":
        registration_object = BSplineNiftyReg
    elif method == "SyNANTs":
        registration_object = SyNANTs
    elif method == "VoxelMorph":
        registration_object = VoxelMorph
    elif method == "DemonsSITK":
        registration_object = DemonsSITK
    elif method == "AffineNiftyReg":
        registration_object = AffineNiftyReg

    registration = registration_object(path_config,
                                       loader_data,
                                       use_masked_evaluation=True)

    #############################
    # with register_dataset()
    #############################
    registration.evaluate_with_zero_displacement()
    registration.execute_with_one_parameter_set()

    #############################
    # with perform_wandb_sweep()
    #############################
    # registration.evaluate_with_zero_displacement()
    # registration.perform_wandb_sweep()


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
