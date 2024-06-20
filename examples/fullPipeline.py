from time import time
from tqdm import tqdm
import matplotlib
from pathlib import Path
import sys
import logging
import socket
import os
import time
import time
import warnings

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
from registrationbaselines.registration.voxelmorph import VoxelmorphReg  # nopep8
from registrationbaselines.registration.bspline_feature import BSplineFeature  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.parent.absolute()

    method = "BSplines"
    # method = "voxelmorph_feature"

    path_config = base_dir / f"registrationbaselines/configs/{method}.yaml"
    config = utils.read_config(path_config)

    if method == "BSplines":
        registration = BSplineFeature(config)
    elif method == "voxelmorph_feature":
        registration = VoxelmorphReg(config)
    else:
        raise ValueError("Method not implemented")

    machine_name = socket.gethostname()
    if machine_name == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/ACDC/")
    elif machine_name == "janus":
        path_data = Path("/data/ACDC/database/")
    else:
        path_data = Path("/home/anna/datasets/FIRE")

    loader_data = data_loaders.ACDCDataset(path_data,
                                           return_mode="val_imgs4",
                                           normalize_mode=True,
                                           roi_only=True,
                                           dim_mode='2d-middle')

    """
    idxs = np.arange(134)
    # np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:124], idxs[124:]
    train_dataset = data_loaders.FIREDataset(
        path_data, return_type="np_arrays_rgb", idxs=[0])
    val_dataset = data_loaders.FIREDataset(
        path_data, return_type="np_arrays_rgb_kps", idxs=[0])
    # train_dataset = MNISTDataset(train=True, subset_range=100, return_type="np_arrays_rgb")
    # val_dataset = MNISTDataset(train=False, subset_range=1, return_type="np_arrays_rgb")
    # loader_data = MNISTDataset(train=False, subset_range=1, return_type="path_dict")

    # train
    # if method == "voxelmorph_feature":
    #     vxm_registration = VoxelmorphFeatureTraining(train_dataset, path_config, val_dataset)
    #     vxm_registration.train()
    """

    registration.register_all_parametr_sets(loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    warnings.filterwarnings('ignore', '.*NiftiImageIO.*',)
    warnings.filterwarnings(
        "ignore", message=".*NiftiImageIO.*unexpected scales in sform")

    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Save execution time to a tex file
    with open('/u/home/koeglf/Documents/code/registrationbaselines/tmp/time.txt', 'w') as f:
        f.write(f"Execution time: {execution_time} seconds")
