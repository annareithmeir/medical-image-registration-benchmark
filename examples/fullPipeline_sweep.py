from time import time
import matplotlib
from pathlib import Path
import sys
import logging
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
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8
from registrationbaselines.registration.syn_ants import SyNANTs  # nopep8


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

    loader_data = data_loaders.L2RLungCTDataset(Path("/home/anna/datasets/LungCT"),
                                                return_type="path_dict",
                                                indices=[0])

    if method == "BSplines":
        registration = BSplineNiftyReg(config, loader_data)
    elif method == "SyNANTs":
        registration = SyNANTs(config, loader_data)
    else:
        raise ValueError("Method not implemented")

    registration.register_all_parametr_sets()


if __name__ == "__main__":
    # Set the logging level for matplotlib to WARNING
    matplotlib.use('Agg')
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
    # with open('/u/home/koeglf/Documents/code/registrationbaselines/tmp/time.txt', 'w') as f:
    # # with open('/u/home/koeglf/Documents/code/registrationbaselines/tmp/time.txt', 'w') as f:
    #     f.write(f"Execution time: {execution_time} seconds")
