from pathlib import Path
import sys
import os
import time
import random

from typing import Dict, Type, Union

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.core import utils  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8
# from registrationbaselines.registration.demons_sitk import DemonsSITK  # nopep8
# from registrationbaselines.registration.lapirn import LapIRN  # nopep8
# from registrationbaselines.registration.syn_ants import SyNANTs  # nopep8
# from registrationbaselines.registration.voxelmorph import VoxelMorph  # nopep8


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.absolute().parent

    # methods: Dict[Type[Union[VoxelMorph, LapIRN, SyNANTs, DemonsSITK, BSplineNiftyReg]], Path] = {
    methods = {
        # VoxelMorph: base_dir / "registrationbaselines/configs/VoxelMorph.yaml",
        # LapIRN: base_dir / "registrationbaselines/configs/LapIRN.yaml",
        # SyNANTs: base_dir / "registrationbaselines/configs/SyNANTs.yaml",
        BSplineNiftyReg: base_dir / "registrationbaselines/configs/BSplineNiftyReg.yaml",
        # DemonsSITK: base_dir / "registrationbaselines/configs/DemonsSITK.yaml",
    }

    models = [
        [  # base_dir / "tmp/displacement_debug/LungCT/VoxelMorph/train/VoxelMorph_comic-sweep-1/model_00500.pt"]  # ,
            base_dir / "tmp/displacement_debug/LungCT/LapIRN/train/LapIRN_splendid-sweep-1/model_level3_final.pt"]

        # [base_dir / "tmp/test_all_trains/AbdomenMRCT/VoxelMorph/train/VoxelMorph_gallant-sweep-1/model_epoch00001_final.pt",
        #  base_dir / "tmp/test_all_trains/AbdomenMRCT/LapIRN/train/LapIRN_valiant-sweep-1/model_level3_final.pt"],

        # [base_dir / "tmp/test_all_trains/image_pairs/VoxelMorph/train/VoxelMorph_a32fedfc-bff3-43a5-afda-c9e0b60523e3/model_epoch00001_final.pt",
        #  base_dir / "tmp/test_all_trains/image_pairs/LapIRN/train/LapIRN_025a7ac9-e1ef-4344-a67c-92c8f9512232/model_level3_final.pt"]
    ]

    indices = random.sample(range(1, 8), 2)
    indices = [0]
    datasets = [
        data_loaders.L2RLungCTDataset(dataset_path=Path("/data/LungCT_preprocessed_anna_test"),
                                      return_type="path_dict",
                                      indices=indices),
        # data_loaders.L2RAbdominalMRCTDataset(dataset_path=Path("/data/AbdomenMRCT_preprocessed_new"),
        #                                      return_type="path_dict",
        #                                      indices=indices),
        # data_loaders.ImagePairDataset(image_pairs=[
        #     [Path(f"/u/home/koeglf/Documents/code/registrationbaselines/registrationbaselines/tests/images/fixed_x_11.nii.gz"),
        #      Path(f"/u/home/koeglf/Documents/code/registrationbaselines/registrationbaselines/tests/images/moving_x_11.nii.gz")]
        # ],
        #     return_type="path_dict")
    ]

    first_dataset_evaluation = True

    for i, dataset in enumerate(datasets):
        first_dataset_evaluation = True

        for j, (method, config_path) in enumerate(methods.items()):
            # get method name
            if method.__name__ in ["VoxelMorph", "LapIRN"]:
                registration = method(config_path,
                                      dataset,
                                      use_masked_evaluation=True,
                                      model_path=models[i][j])
            else:
                registration = method(config_path,
                                      dataset,
                                      use_masked_evaluation=True)

            if first_dataset_evaluation and dataset.name != "image_pairs":
                # registration.evaluate_with_zero_displacement()
                first_dataset_evaluation = False

            registration.execute_with_one_parameter_set()

            # if not hasattr(registration, "model_path"):
            #     registration.perform_wandb_sweep()


if __name__ == "__main__":

    utils.turn_off_warnings()

    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
