from pathlib import Path
import sys
import os
import time

from typing import Optional, List, Tuple

import wandb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc

from registrationbaselines.interfaces._interface_training import TrainingInterface
from registrationbaselines.core import utils_dl
from registrationbaselines.data_loading import data_loaders
import icon_registration as icon_registration  # nopep8
import icon_registration.networks as networks  # nopep8
import icon_registration.network_wrappers as network_wrappers  # nopep8

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8


gc.collect()
torch.cuda.empty_cache()


class ICON(TrainingInterface):
    """
    Training for ICON.
    """

    def __init__(self,
                 train_dataset: data_loaders.GenericDataset,
                 configuration_path: Path,
                 val_dataset: Optional[data_loaders.GenericDataset] = None) -> None:

        super().__init__("ICON",
                         configuration_path,
                         train_dataset,
                         val_dataset)

    def _train(self) -> None:

        self.network = self._make_network(
            [1, 1] + list(self.train_dataset.image_shape))

        self.optimizer = torch.optim.Adam(
            (p for p in self.network.parameters() if p.requires_grad), lr=0.00005
        )

        generator_train = self._batch_generator(self.train_dataset)
        generator_val = None if self.val_dataset is None else self._batch_generator(
            self.val_dataset)

        for epoch in range(self.run_configuration['epochs']):

            self.network.train()
            self.optimizer.zero_grad()

            if epoch != 0 and epoch % self.run_configuration['save_checkpoint'] == 0:
                self._save_current_state(epoch)

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for moving_image, fixed_image in generator_train:

                loss_object = self.network(moving_image, fixed_image)

                loss = torch.mean(loss_object.all_loss)

                loss.backward()

                # print(to_floats(loss_object))

                self.optimizer.step()

            if generator_val is not None:
                self.network.eval()

                for moving_image, fixed_image in generator_val:

                    with torch.no_grad():
                        loss = self.network(moving_image, fixed_image)

                self.network.train()

    def _batch_generator(self, dataset) -> Tuple[torch.Tensor, torch.Tensor]:

        # maybe do this in each epoch, so we always get a shuffle
        dataloader = DataLoader(dataset,
                                batch_size=self.run_configuration['batch_size'],
                                shuffle=True)

        for item in dataloader:
            fixed_image = item['fixed_image'].cuda()
            moving_image = item['moving_image'].cuda()

            fixed_image = utils_dl.reshape_tensor(
                fixed_image, [4 * 40, 4 * 96, 4 * 96])
            moving_image = utils_dl.reshape_tensor(
                moving_image, [4 * 40, 4 * 96, 4 * 96])

            fixed_image.unsqueeze_(0).unsqueeze_(0)
            moving_image.unsqueeze_(0).unsqueeze_(0)

            yield moving_image, fixed_image

    def _make_network(self, shape: List[int]):

        phi = network_wrappers.FunctionFromVectorField(
            networks.tallUNet2(dimension=3)
        )
        psi = network_wrappers.FunctionFromVectorField(
            networks.tallUNet2(dimension=3))

        hires_net = icon_registration.GradientICON(
            network_wrappers.DoubleNet(
                network_wrappers.DownsampleNet(
                    network_wrappers.TwoStepRegistration(phi, psi), dimension=3
                ),
                network_wrappers.FunctionFromVectorField(
                    networks.tallUNet2(dimension=3)),
            ),
            icon_registration.LNCCOnlyInterpolated(sigma=5),
            3,
        )
        hires_net.assign_identity_map([1, 1, 4 * 40, 4 * 96, 4 * 96])
        return hires_net

    def _save_current_state(self, epoch: int) -> None:
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.path_dir_run, f"optimizer_{epoch:05d}.pt")
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.path_dir_run, f"model_{epoch:05d}.pt")
        )
