from pathlib import Path
import sys
import os
import time

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.core import utils  # nopep8


def main() -> None:
    """
    Main function to run the full registration and evaluation pipeline.
    """

    # dataset = data_loaders.L2RLungCTDataset(dataset_path=Path("/data/LungCT"),
    #                                         return_type="path_dict")

    # dataset.preprocess(Path("/data/LungCT_preprocessed_new"))

    dataset = data_loaders.L2RAbdominalMRCTDataset(dataset_path=Path("/data/AbdomenMRCT"),
                                                   return_type="path_dict")

    dataset.preprocess(Path("/data/AbdomenMRCT_preprocessed_new"))


if __name__ == "__main__":

    utils.turn_off_warnings()

    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
