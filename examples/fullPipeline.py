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

from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    path_config = Path(__file__).parent.parent.absolute() / \
        f"registrationbaselines/configs/BSplineNiftyReg.yaml"

    if socket.gethostname() == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/LungCT_preprocessed/")
    elif socket.gethostname() == "janus":
        path_data = Path("/data/LungCT_preprocessed")
    else:
        path_data = Path("/home/anna/datasets/LungCT_preprocessed")

    loader_data = data_loaders.ImagePairDataset([[Path("/home/fryderyk/Documents/data/LungCT_preprocessed/imagesTr/LungCT_0001_0000.nii.gz"),
                                                  Path("/home/fryderyk/Documents/data/LungCT_preprocessed/imagesTr/LungCT_0001_0001.nii.gz")]],
                                                return_type="path_dict")

    registration = BSplineNiftyReg(path_config, loader_data)

    # Register a dataset - can be image pair dataset
    # registration.register_dataset()

    # WANDB SWEEP
    # registration.perform_wandb_sweep()

    ######
    loader_data = data_loaders.L2RLungCTDataset(path_data,
                                                return_type="path_dict",
                                                indices=[1, 2])

    registration = BSplineNiftyReg(path_config,
                                   loader_data)

    # Register a dataset - can be image pair dataset
    registration.register_dataset()

    # WANDB SWEEP
    registration.perform_wandb_sweep()


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
