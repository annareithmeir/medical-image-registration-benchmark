import os
import unittest
from pathlib import Path
import sys

import torch.cuda
import wandb
import numpy as np
sys.path.append(str(Path(__file__).parent.absolute().parent.parent))



from registrationbaselines.data_loading.data_loaders import DemoImageDataset, L2RLungCTDataset


class TestVoxelmorphTraining(unittest.TestCase):

    def test_voxelmorph_training_L2RLunGCT_train_val_split(self):
        from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
        idxs = np.arange(20)
        np.random.shuffle(idxs)
        train_idx, val_idx = idxs[:15], idxs[15:]
        print(train_idx, val_idx)
        train_dataset = L2RLungCTDataset(imgs_path=self.lung_dataset_path,
                                   transforms=["normalize"], idxs=list(train_idx))
        val_dataset = L2RLungCTDataset(imgs_path=self.lung_dataset_path,
                                   transforms=["normalize"], idxs=list(val_idx))
        self.assertEqual(len(train_dataset), 15)
        self.assertEqual(len(val_dataset), 5)

    def test_voxelmorph_training_L2RLunGCT_wandb(self):
        from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
        idxs = np.arange(20)
        np.random.shuffle(idxs)
        train_idx, val_idx = idxs[:15], idxs[15:]
        #print(train_idx, val_idx)
        base_dir = Path(__file__).parent.parent.absolute().parent
        train_dataset = L2RLungCTDataset(imgs_path=self.lung_dataset_path,
                                         transforms=["normalize"], idxs=list(train_idx))
        val_dataset = L2RLungCTDataset(imgs_path=self.lung_dataset_path,
                                       transforms=["normalize"], idxs=list(val_idx))

        print("train dataset:", len(train_dataset), " val dataset: ", len(val_dataset))

        vxm_config_file = base_dir / "registrationbaselines/configs/TrainVoxelmorph.yaml"
        vxm_training = VoxelmorphTraining(train_dataset, vxm_config_file, val_dataset)
        vxm_training.train()

    def test_gpu(self):
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())


class TestLapirnTraining(unittest.TestCase):

    def test_lapirn_training(self):
        from registrationbaselines.training.train_lapirn import LapIRNTraining
        idxs = np.arange(20)
        np.random.shuffle(idxs)
        train_idx, val_idx = idxs[:15], idxs[15:]
        # print(train_idx, val_idx)
        base_dir = Path(__file__).parent.parent.absolute().parent
        train_dataset = L2RLungCTDataset(imgs_path=Path("/home/anna/datasets/LungCT"),
                                         transforms=["normalize"], idxs=list(train_idx))
        val_dataset = L2RLungCTDataset(imgs_path=Path("/home/anna/datasets/LungCT"),
                                       transforms=["normalize"], idxs=list(val_idx))

        print("train dataset:", len(train_dataset), " val dataset: ", len(val_dataset))

        lapirn_config_file = base_dir / "registrationbaselines/configs/TrainLapirn.yaml"
        lapirn_training = LapIRNTraining(train_dataset, lapirn_config_file, val_dataset)
        lapirn_training.train()



if __name__ == '__main__':
    unittest.main()

