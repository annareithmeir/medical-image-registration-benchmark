import os
import unittest
from pathlib import Path
from torch.utils.data import DataLoader
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent.parent))

from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
from registrationbaselines.data_loading.data_loaders import DemoImageDataset, L2RLungCTDataset
from registrationbaselines.core.train_configurations import VoxelmorphTrainConfiguration

class TestDataloaders(unittest.TestCase):

    def test_train_dataloader(self):
        base_dir = Path(__file__).parent.parent.absolute().parent
        dataset = DemoImageDataset(imgs_path=base_dir / "registrationbaselines/data/training_dataset", transforms=["clip_bones"])
        self.assertEqual(len(dataset), 3)

        train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        ins, outs = next(iter(train_dataloader))
        self.assertEqual(ins.shape, (2, 1) + dataset.img_shape)

    def test_plot_random_subject(self):
        base_dir = Path(__file__).parent.parent.absolute().parent
        dataset = DemoImageDataset(imgs_path=base_dir / "registrationbaselines/data/training_dataset", transforms=["clip_bones"])
        dataset.plot_random_image()

    def test_get_item(self):
        base_dir = Path(__file__).parent.parent.absolute().parent
        dataset = DemoImageDataset(imgs_path=base_dir / "registrationbaselines/data/training_dataset", transforms=["clip_bones"])
        m, f = dataset.__getitem__(0)
        self.assertEqual(m.shape[1:], dataset.img_shape)
        print(m.shape)

    def test_vxm_scan_to_scan_generator(self):
        base_dir = Path(__file__).parent.parent.absolute().parent
        dataset = DemoImageDataset(imgs_path=base_dir / "registrationbaselines/data/training_dataset",
                                   transforms=["clip_bones"])
        vxm_config = VoxelmorphTrainConfiguration(
            result_model_path= base_dir / "tmp/test.pt", epochs=1,
            steps_per_epoch=1, batch_size=2)
        vxm_training = VoxelmorphTraining(dataset, vxm_config)

        gen = vxm_training.scan_to_scan_generator()
        ins,outs=next(iter(gen))
        self.assertEqual(ins[0].shape, (2, 1) + dataset.img_shape) #m (bs,1,h,w,d)
        self.assertEqual(ins[1].shape, (2, 1) + dataset.img_shape) #f (bs,1,h,w,d)
        self.assertEqual(outs[1].shape, (2, 3) + dataset.img_shape) #zeros (bs,3,h,w,d)


    def test_L2RLungCTDataset(self):
        dataset = L2RLungCTDataset(imgs_path=Path("/home/anna/datasets/LungCT"),
                                   transforms=["normalize", "resample"])
        self.assertEqual(len(dataset), 20)
        dataset.plot_random_image()

        m, f = dataset.__getitem__(0)
        self.assertEqual(m.shape[1:], dataset.img_shape)
        self.assertEqual(dataset.spacing, (1,1,1))




if __name__ == '__main__':
    unittest.main()


