import os
import unittest
from pathlib import Path
import torch

import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent.parent))

from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset, L2RAbdominalMRCTDataset

class TestDataloaders(unittest.TestCase):

    def test_L2RLungCTDataset(self):
        dataset = L2RLungCTDataset(imgs_path=Path("/home/anna/datasets/LungCT"),
                                   transforms=["normalize", "resample"],
                                   return_type="np_array")
        self.assertEqual(len(dataset), 20)

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        self.assertEqual(m.shape[1:], dataset.img_shape) # shape[0] is batchsize

        dataset.preprocess(Path("/home/anna/LungCT_preprocessed"))

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        self.assertEqual(m.shape[1:], dataset.img_shape)  # shape[0] is batchsize
        self.assertEqual(dataset.spacing, (1.75, 1.75, 1.75))

        dataset.plot_random_image()

    def test_L2RLungCTDataset_idxs(self):
        dataset = L2RLungCTDataset(imgs_path=Path("/home/anna/datasets/LungCT"),
                                   return_type="np_array",
                                   idxs=[0])
        self.assertEqual(len(dataset), 1)

    def test_L2RAbdominalMRCTDataset(self):
        # dataset = L2RAbdominalMRCTDataset(imgs_path=Path("/home/anna/datasets/AbdomenMRCT"),
        #                            transforms=["normalize"],
        #                            return_type="np_array_dict")
        dataset = L2RAbdominalMRCTDataset(dataset_path=Path("/home/anna/AbdomenMRCT_preprocessed"),
                                          return_type="np_array_dict")
        self.assertEqual(len(dataset), 8)

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        self.assertEqual(m.shape[1:], dataset.images_shape) # shape[0] is batchsize

        # dataset.preprocess(Path("/home/anna/AbdomenMRCT_preprocessed"))

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        self.assertEqual(m.shape[1:], dataset.images_shape)  # shape[0] is batchsize
        self.assertEqual(dataset.spacing, (2,2,2))

        dataset.plot_random_image()




if __name__ == '__main__':
    unittest.main()


