from pathlib import Path
import sys
import logging
import os
import time
import random

from typing import Dict, Type, Union

import SimpleITK as sitk

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8
from registrationbaselines.registration.affine_niftyreg import AffineNiftyReg  # nopep8
from registrationbaselines.registration.voxelmorph import VoxelMorph  # nopep8
from registrationbaselines.registration.syn_ants import SyNANTs  # nopep8
from registrationbaselines.registration.demons_sitk import DemonsSITK  # nopep8


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    base_dir = Path(__file__).parent.absolute().parent

    methods: Dict[Type[Union[SyNANTs, DemonsSITK]], Path] = {
        SyNANTs: base_dir / "registrationbaselines/configs/SyNANTs.yaml",
        DemonsSITK: base_dir / "registrationbaselines/configs/DemonsSITK.yaml",
    }

    indices = random.sample(range(1, 9), 2)
    datasets = [
        data_loaders.L2RLungCTDataset(dataset_path=Path("/data/LungCT_preprocessed"),
                                      return_type="path_dict",
                                      indices=indices),
        data_loaders.L2RAbdominalMRCTDataset(dataset_path=Path("/data/AbdomenMRCT_preprocessed"),
                                             return_type="path_dict",
                                             indices=indices),
        data_loaders.ImagePairDataset(image_pairs=[
            [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{indices[0]:02}_0000.nii.gz"),
             Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{indices[0]:02}_0001.nii.gz")]
        ],
            return_type="path_dict")
    ]

    first_dataset_evaluation = True

    for dataset in datasets:
        first_dataset_evaluation = True

        for method, config_path in methods.items():

            registration = method(config_path,
                                  dataset,
                                  use_masked_evaluation=True)

            if first_dataset_evaluation and dataset.name != "image_pairs":
                registration.evaluate_with_zero_displacement()
                first_dataset_evaluation = False

            registration.execute_with_one_parameter_set()

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
