import os
import unittest
from pathlib import Path
import torch

import numpy as np
from torch.utils.data import DataLoader
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset, L2RAbdominalMRCTDataset, L2RAbdominalCTCTDataset


class TestDataloaders(unittest.TestCase):

    def test_L2RLungCTDataset(self):
        dataset = L2RLungCTDataset(dataset_path=Path("/home/anna/datasets/LungCT"),
                                   transforms=["normalize", "resample"],
                                   return_type="np_array")
        self.assertEqual(len(dataset), 20)

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        # shape[0] is batchsize
        self.assertEqual(m.shape[1:], dataset.images_shape)

        dataset.preprocess(Path("/home/anna/LungCT_preprocessed"))

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        # shape[0] is batchsize
        self.assertEqual(m.shape[1:], dataset.images_shape)
        self.assertEqual(dataset.spacing, (1.75, 1.75, 1.75))

        dataset.plot_random_image()

    def test_L2RLungCTDataset_idxs(self):
        dataset = L2RLungCTDataset(dataset_path=Path("/home/anna/datasets/LungCT"),
                                   return_type="np_array",
                                   indices=[0])
        self.assertEqual(len(dataset), 1)

    def test_L2RAbdominalMRCTDataset(self):
        dataset = L2RAbdominalMRCTDataset(dataset_path=Path("/home/anna/datasets/AbdomenMRCT"),
                                          transforms=["normalize"],
                                          return_type="np_array_dict",
                                          )
        # dataset = L2RAbdominalMRCTDataset(dataset_path=Path("/home/anna/AbdomenMRCT_preprocessed"),
        #                                   return_type="np_array_dict")
        self.assertEqual(len(dataset), 8)

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        # shape[0] is batchsize
        self.assertEqual(m.shape[1:], dataset.images_shape)

        dataset.preprocess(
            Path("/home/anna/datasets/AbdomenMRCT_preprocessed"))

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        # shape[0] is batchsize
        self.assertEqual(m.shape[1:], dataset.images_shape)
        self.assertEqual(dataset.spacing, (2, 2, 2))

        dataset.plot_random_image()

    def test_L2RAbdominalCTDataset(self):
        # dataset = L2RAbdominalCTCTDataset(dataset_path=Path("/home/anna/datasets/AbdomenCTCT"),
        #                            transforms=["normalize"],
        #                            return_type="np_array_dict")
        dataset = L2RAbdominalCTCTDataset(dataset_path=Path("/home/anna/AbdomenCTCT_preprocessed"),
                                          return_type="np_array_dict")
        self.assertEqual(len(dataset), 435)

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        # shape[0] is batchsize
        self.assertEqual(m.shape[1:], dataset.images_shape)

        # dataset.preprocess(Path("/home/anna/datasets/AbdomenCTCT_preprocessed"))

        item = dataset.__getitem__(0)
        m = item["images"][0]
        self.assertIsInstance(m, torch.Tensor)
        # shape[0] is batchsize
        self.assertEqual(m.shape[1:], dataset.images_shape)
        self.assertEqual(dataset.spacing, (2, 2, 2))

        dataset.plot_random_image()


if __name__ == '__main__':
    unittest.main()
