import os
import unittest
from pathlib import Path
import sys
import wandb
sys.path.append(str(Path(__file__).parent.absolute().parent.parent))

from registrationbaselines.training.train_voxelmorph import VoxelmorphTraining
from registrationbaselines.data_loading.data_loaders import DemoImageDataset
from registrationbaselines.core.train_configurations import VoxelmorphTrainConfiguration
from registrationbaselines.core.train_configurations import WandbConfiguration

class TestDLModelsTraining(unittest.TestCase):

    def test_voxelmorph_training(self):
        base_dir = Path(__file__).parent.parent.absolute().parent
        dataset = DemoImageDataset(imgs_path=base_dir / "registrationbaselines/data/training_dataset", transforms=["clip_bones"])
        vxm_config = VoxelmorphTrainConfiguration(result_model_path="/home/anna/PycharmProjects/registrationbaselines/tmp/test",initial_weights_path="/home/anna/PycharmProjects/registrationbaselines/tmp/test/_w.pt", epochs=3, steps_per_epoch=1)
        vxm_training = VoxelmorphTraining(dataset, vxm_config)
        vxm_training.train()

    def test_wandb(self):
        wandb_config = WandbConfiguration(project="voxelmorph",name="test")

        wandb.init(
            project = wandb_config.project,
            group=wandb_config.group,
            name=wandb_config.name,
            config=wandb_config.config_dict
        )

        #wandb.log({"validation": table})
        wandb.finish()

    def test_voxelmorph_training_wandb(self):
        base_dir = Path(__file__).parent.parent.absolute().parent
        dataset = DemoImageDataset(imgs_path=base_dir / "registrationbaselines/data/training_dataset",
                                   transforms=["clip_bones"])
        wandb_config = WandbConfiguration(project="voxelmorph", name="test2")
        vxm_config = VoxelmorphTrainConfiguration(
            result_model_path="/home/anna/PycharmProjects/registrationbaselines/tmp/test",
            initial_weights_path="/home/anna/PycharmProjects/registrationbaselines/tmp/test/_w.pt", epochs=30,
            steps_per_epoch=1,
            use_wandb=True,
            wandb_config=wandb_config)
        vxm_training = VoxelmorphTraining(dataset, vxm_config)
        vxm_training.train()


if __name__ == '__main__':
    unittest.main()

