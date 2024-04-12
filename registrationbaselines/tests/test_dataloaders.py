import unittest
from registrationbaselines.data_loading.data_loaders import DemoImageDataset
from pathlib import Path
import sys

class TestDataloaders(unittest.TestCase):

    def test_train_dataloader(self):
        dataset = DemoImageDataset(imgs_path=Path("registrationbaselines/data/training_dataset"))
        self.assertEqual(len(dataset), 3)

    def test_plot_random_subject(self):
        dataset = DemoImageDataset(imgs_path=Path("registrationbaselines/data/training_dataset"), transforms=["clip_bones"])
        dataset.plot_random_image()


if __name__ == '__main__':
    unittest.main()
