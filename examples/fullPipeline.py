from pathlib import Path
import sys
import logging
import socket
import os
# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import matplotlib
import numpy as np

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg
from registrationbaselines.registration.bspline_feature import BSplineFeature
from registrationbaselines.registration.voxelmorph import VoxelmorphReg
from registrationbaselines.training.train_voxelmorph_feature import VoxelmorphFeatureTraining
# from registrationbaselines.registration.convexadam import ConvexAdam
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders

sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8

# from latent_space_registration.datasets import MNISTDataset


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.parent.absolute()

    # method = "BSplines"
    # method = "convexAdam"
    # method = "BSplineMedSAM"
    method = "voxelmorph_feature"

    if method == "BSplines":
        path_config = base_dir / 'registrationbaselines/configs/BSplineNiftyReg.yaml'
        registration = BSplineNiftyReg(path_config)
    elif method == "convexAdam":
        path_config = base_dir / 'registrationbaselines/configs/ConvexAdam.yaml'
        # registration = ConvexAdam(path_config)
    elif method == "BSplineMedSAM":
        path_config = base_dir / 'registrationbaselines/configs/BSplineMedSAM.yaml'
        registration = BSplineFeature(path_config)
    elif method == "voxelmorph_feature":
        path_config = base_dir / 'registrationbaselines/configs/Voxelmorph_feature.yaml'
        registration = VoxelmorphReg(path_config)
    else:
        print("Method not implemented")

    machine_name = socket.gethostname()
    if machine_name == "fryderyk":
        path_data = Path("/home/fryderyk/Documents/data/LungCT")
    elif machine_name == "janus":
        path_data = Path("/u/home/koeglf/Documents/data/LungCT")
        path_data = Path("/data/FIRE/")
    else:
        path_data = Path("/home/anna/datasets/ACDC")
        # path_data = Path("/home/anna/datasets/FIRE")
        # path_data = Path("/home/anna/datasets/AbdomenCTCT_preprocessed")
        # path_data = Path("/home/anna/datasets/LungCT")

    # indices = [0]
    # loader_data = data_loaders.L2RAbdominalCTCTDataset(
    #     path_data, indices=indices)
    # loader_data = data_loaders.L2RLungCTDataset(path_data, indices = indices)

    idxs = np.arange(134)
    #np.random.shuffle(idxs)
    trafos=["normalize", "greyscale"]
    train_idx, val_idx = idxs[:124], idxs[124:]
    idx = 2
    # loader_data = data_loaders.FIREDataset(path_data, return_type="path_dict", idxs=[idx], transforms=trafos)
    # train_dataset = data_loaders.FIREDataset(path_data, return_type="np_arrays_rgb", idxs=[idx], transforms=trafos)
    # val_dataset = data_loaders.FIREDataset(path_data, return_type="np_arrays_rgb_kps",idxs=[idx], transforms=trafos)
    # train_dataset = MNISTDataset(train=True, subset_range=100, return_type="np_arrays_rgb")
    # val_dataset = MNISTDataset(train=False, subset_range=1, return_type="np_arrays_rgb")
    # loader_data = MNISTDataset(train=False, subset_range=1, return_type="path_dict")

    #ACDC retrun mode is "<2/4>_<train/val/test>". 2 only gives images, 4 also gives labels
    train_dataset = data_loaders.ACDCDataset(path_data, return_mode = "train_imgs4", normalize_mode=True, roi_only=True, dim_mode='2d-middle', idxs=[0])
    val_dataset = data_loaders.ACDCDataset(path_data, return_mode = "val_imgs4", normalize_mode=True, roi_only=True, dim_mode='2d-middle', idxs=[0])
    train_dataset.plot_random_image()

    # train
    if method == "voxelmorph_feature":
        vxm_registration = VoxelmorphFeatureTraining(train_dataset, path_config, val_dataset)
        vxm_registration.train()

    # register
    # print("\nregister...")
    # for i in tqdm(range(len(loader_data))):
    #     item = loader_data[i]
    #     registration.register(item["images"][0], item["images"][1])
    #     if i == 1:
    #         break
    #
    # if method == "BSplines":
    #     loader_transformations = data_loaders.BaselineTransformations(
    #         base_dir / "tmp/AbdomenCTCT/BSplineNiftyReg")
    # elif method == "convexAdam":
    #     loader_transformations = data_loaders.BaselineTransformations(
    #         base_dir / "tmp/AbdomenCTCT/ConvexAdam")
    # elif method == "BSplineMedSAM":
    #     loader_transformations = data_loaders.BaselineTransformations(
    #         base_dir / "tmp/results/BSplineMedSAM")
    # else:
    #     print("Method not implemented")
    #
    # # evaluate
    # print("\nevaluate...")
    # evaluation = Evaluation(path_config)
    # evaluation.evaluate(loader_transformations, loader_data)
    # print("\nplot...")
    # evaluation.visualize(loader_transformations, loader_data)


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    # matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    main()
