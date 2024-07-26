from pathlib import Path
import sys


import numpy as np

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset


def main():

    idxs = np.arange(5)
    np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:4], idxs[4:]

    base_dir = Path(__file__).parent.parent.absolute()

    data_path = Path("/data/LungCT_preprocessed")

    train_dataset = L2RLungCTDataset(dataset_path=data_path,
                                     indices=list(train_idx),
                                     return_type="torch_tensor_dict")
    val_dataset = L2RLungCTDataset(dataset_path=data_path,
                                   indices=list(val_idx),
                                   return_type="torch_tensor_dict")

    print("train dataset:", len(train_dataset),
          "val dataset:  ", len(val_dataset))

    vxm_config_file = base_dir / "registrationbaselines/configs/Voxelmorph.yaml"
    vxm_training = VoxelmorphTraining(
        train_dataset, vxm_config_file, val_dataset)

    vxm_training.train()


if __name__ == "__main__":
    main()
