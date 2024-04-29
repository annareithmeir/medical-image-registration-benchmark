from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset


idxs = np.arange(5)
np.random.shuffle(idxs)
train_idx, val_idx = idxs[:15], idxs[15:]

base_dir = Path(__file__).parent.parent.absolute().parent

train_dataset = L2RLungCTDataset(imgs_path=Path("/home/fryderyk/Documents/data/LungCT"),
                                transforms=["normalize"], idxs=list(train_idx))
val_dataset = L2RLungCTDataset(imgs_path=Path("/home/fryderyk/Documents/data/LungCT"),
                            transforms=["normalize"], idxs=list(val_idx))

print("train dataset:", len(train_dataset), " val dataset: ", len(val_dataset))

vxm_config_file = base_dir / "registrationbaselines/configs/TrainVoxelmorph.yaml"
vxm_training = VoxelmorphTraining(train_dataset, vxm_config_file, val_dataset)

vxm_training.train()
