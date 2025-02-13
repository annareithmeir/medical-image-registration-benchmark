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

"""
for epoch:
    epoch_loss = [] -> list of lists, small lists are sim and grad loss
    epoch_total_loss = [] -> list of total oss (sim + grad)
    epoch_step_time = []

    for step
        loss = 0 -> sim loss + grad loss
        loss_list = [] -> sim loss and grad loss


        same for validation

    val_loss_list = []

"""


class GradICON(TrainingInterface):
    """
    Training for ICON.
    """

    def __init__(self,
                 train_dataset: data_loaders.GenericDataset,
                 configuration_path: Path,
                 val_dataset: Optional[data_loaders.GenericDataset] = None) -> None:

        super().__init__("GradICON",
                         configuration_path,
                         train_dataset,
                         val_dataset)

        self._multiple_gpus = False

    def _train(self) -> None:

        gpus = list(map(int, self.run_configuration['gpu'].split(',')))
        nb_gpus = len(gpus)

        assert np.mod(self.run_configuration['batch_size'], nb_gpus) == 0, \
            'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (
            self.run_configuration['batch_size'], nb_gpus)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            map(str, gpus))  # Ensure GPUs are set

        self.network = utils_dl.make_gradicon_network([1, 1] + list(self.train_dataset.image_shape),
                                                      self.run_configuration['batch_size'])

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.network = torch.nn.DataParallel(self.network, device_ids=gpus)

            self._multiple_gpus = True

        self.optimizer = torch.optim.Adam(
            (p for p in self.network.parameters() if p.requires_grad), lr=self.run_configuration['lr']
        )

        for epoch in range(self.run_configuration['epochs']):
            generator_train = self._batch_generator(self.train_dataset)
            generator_val = None if self.val_dataset is None else self._batch_generator(
                self.val_dataset)

            self.network.train()
            self.optimizer.zero_grad()

            if epoch != 0 and epoch % self.run_configuration['save_checkpoint'] == 0:
                self._save_current_state(epoch)

            epoch_loss = []
            epoch_total_loss = []

            for moving_image, fixed_image in generator_train:

                loss_object = self.network(moving_image, fixed_image)

                loss = torch.mean(loss_object.all_loss)

                loss.backward()

                self.optimizer.step()

                epoch_total_loss.append(loss.item())
                epoch_loss.append([torch.sum(loss_object.inverse_consistency_loss).item(),
                                   torch.sum(loss_object.similarity_loss).item()])

            if generator_val is not None:
                val_loss = []

                self.network.eval()

                for moving_image, fixed_image in generator_val:

                    with torch.no_grad():
                        loss = self.network(moving_image, fixed_image)

                    val_loss.append([torch.sum(loss.inverse_consistency_loss).item(),
                                     torch.sum(loss.similarity_loss).item()])

                self.network.train()

                # wandb logging
            if self.use_wandb:
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(epoch_total_loss),
                               "inv-cons-loss": np.mean(epoch_loss, axis=0)[0],
                               "sim-loss": np.mean(epoch_loss, axis=0)[1],
                               "val-loss": np.mean(val_loss)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss),
                               "inv-cons-loss": np.mean(epoch_loss, axis=0)[0],
                               "sim-loss": np.mean(epoch_loss, axis=0)[1]})

    def _batch_generator(self, dataset) -> Tuple[torch.Tensor, torch.Tensor]:

        def custom_collate(batch):
            fixed_images = torch.stack(
                [item['fixed_image'].unsqueeze(0) for item in batch], dim=0)
            moving_images = torch.stack(
                [item['moving_image'].unsqueeze(0) for item in batch], dim=0)

            return {"fixed_image": fixed_images, "moving_image": moving_images}

        # maybe do this in each epoch, so we always get a shuffle
        dataloader = DataLoader(dataset,
                                batch_size=self.run_configuration['batch_size'],
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=custom_collate)

        for item in dataloader:
            fixed_image = item['fixed_image'].cuda()
            moving_image = item['moving_image'].cuda()

            bs = self.run_configuration['batch_size']
            fixed_image = utils_dl.reshape_tensor(
                fixed_image, [bs, 1, 4 * 40, 4 * 96, 4 * 96])
            moving_image = utils_dl.reshape_tensor(
                moving_image, [bs, 1, 4 * 40, 4 * 96, 4 * 96])

            yield moving_image, fixed_image

    def _save_current_state(self, epoch: int) -> None:
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.path_dir_run, f"optimizer_{epoch:05d}.pt")
        )
        torch.save(
            self.network.module.state_dict() if self._multiple_gpus else self.network.state_dict(),
            os.path.join(self.path_dir_run, f"model_{epoch:05d}.pt")
        )
