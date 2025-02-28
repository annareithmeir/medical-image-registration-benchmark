import math
from pathlib import Path
import sys
import os
import time

from typing import Optional

import wandb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
import torch.nn.functional as F

from registrationbaselines.interfaces._interface_training import TrainingInterface
from registrationbaselines.core import utils_dl
import registrationbaselines.dl_repos.voxelmorph.voxelmorph as vxm
from registrationbaselines.data_loading import data_loaders

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8


gc.collect()
torch.cuda.empty_cache()


class VoxelMorph(TrainingInterface):
    """
    Training for voxelmorph.
    """

    def __init__(self,
                 train_dataset: data_loaders.GenericDataset,
                 configuration_path: Path,
                 path_results: Path,
                 val_dataset: Optional[data_loaders.GenericDataset] = None,
                 use_logger=False) -> None:

        super().__init__(method_name="VoxelMorph",
                         configuration_path=configuration_path,
                         path_results=path_results,
                         train_dataset=train_dataset,
                         val_dataset=val_dataset,
                         use_logger=use_logger)

        new_shape = utils_dl.get_new_voxelmorph_image_shape(self.train_dataset.image_shape,
                                                            len(self.general_configuration["parameters"]["enc"]["values"][0]))

        self.train_dataset.image_shape = new_shape
        if self.val_dataset:
            self.val_dataset.image_shape = new_shape

    def scan_to_scan_generator(self, dataset: Dataset):
        """
        Reimplementation from vxm.generators.py to fit with dataset class
        (voxelmorph uses an internal generator that prepares the images and an 'empty' deformation in lists for inputs and outputs
        The basis generator for the desired dataset from the original voxelmorph code
        invols = [x, y] both of shape (bs, 1, h,d,w)
        outvols = [y, zeros] zeros of shape (bs,3,h,d,w)

        :return: Generator with data of form (invols[m,f], outvols[m,f])
        """

        def custom_collate(batch):
            fixed_images = torch.stack(
                [item['fixed_image'].unsqueeze(0) for item in batch], dim=0)
            moving_images = torch.stack(
                [item['moving_image'].unsqueeze(0) for item in batch], dim=0)

            return {"fixed_image": fixed_images, "moving_image": moving_images}

        dataloader = DataLoader(dataset,
                                batch_size=self.run_configuration['batch_size'],
                                shuffle=True,
                                pin_memory=True,  # Helps with GPU memory transfer
                                num_workers=8,   # Use more workers to avoid bottlenecks)
                                collate_fn=custom_collate)
        while True:
            item = next(iter(dataloader))

            y = item['fixed_image']
            x = item['moving_image']

            # x = F.interpolate(x, size=self.train_dataset.image_shape,
            #                   mode='trilinear', align_corners=True)
            # y = F.interpolate(y, size=self.train_dataset.image_shape,
            #                   mode='trilinear', align_corners=True)

            # y = utils_dl.pad_tensor_to_shape(y, self.train_dataset.image_shape)
            # x = utils_dl.pad_tensor_to_shape(x, self.train_dataset.image_shape)

            shape = x.shape[2:]
            zeros = torch.from_numpy(
                np.zeros((self.run_configuration['batch_size'], len(shape), *shape)))

            invols = [x, y]
            outvols = [y, zeros]
            yield (invols, outvols)

    def _train(self) -> None:

        # scan-to-scan generator
        generator = self.scan_to_scan_generator(self.train_dataset)
        if self.val_dataset is not None:
            val_generator = self.scan_to_scan_generator(self.val_dataset)

        # extract shape from sampled input
        inshape = self.train_dataset.image_shape

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # device handling
        # Convert to list of integers
        gpus = list(map(int, self.run_configuration['gpu'].split(',')))
        nb_gpus = len(gpus)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            map(str, gpus))  # Ensure GPUs are set

        assert np.mod(self.run_configuration['batch_size'], nb_gpus) == 0, \
            'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (
                self.run_configuration['batch_size'], nb_gpus)

        # enabling cudnn determinism appears to speed up training by a lot
        torch.backends.cudnn.deterministic = not self.run_configuration['cudnn_nondet']

        # unet architecture
        enc_nf = self.run_configuration['enc']
        dec_nf = self.run_configuration['dec']

        if self.run_configuration['load_model']:
            # load initial model (if specified)
            model = vxm.networks.VxmDense.load(
                self.run_configuration['load_model'], device)
        else:
            # otherwise configure new model
            model = vxm.networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=self.run_configuration['bidir'],
                int_steps=self.run_configuration['int_steps'],
                int_downsize=self.run_configuration['int_downsize']
            )

        if nb_gpus > 1:
            # use multiple GPUs via DataParallel
            model = torch.nn.DataParallel(model, device_ids=gpus)
            model.save = model.module.save

        # prepare the model for training and send to device
        model.to(device)
        self.model = model
        self.save_initial_weights()
        model.train()

        # set optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.run_configuration['lr'])

        # prepare image loss
        if self.run_configuration['sim_loss'] == 'ncc':
            image_loss_func = vxm.losses.NCC().loss
        elif self.run_configuration['sim_loss'] == 'mse':
            image_loss_func = vxm.losses.MSE().loss
        else:
            raise ValueError(
                'Image loss should be "mse" or "ncc", but found "%s"' % self.run_configuration['image_loss'])

        # need two image loss functions if bidirectional
        if self.run_configuration['bidir']:
            losses = [image_loss_func, image_loss_func]
            weights = [0.5, 0.5]
        else:
            losses = [image_loss_func]
            weights = [1]

        # prepare deformation loss
        losses += [vxm.losses.Grad('l2',
                                   loss_mult=self.run_configuration['int_downsize']).loss]
        weights += [self.run_configuration['reg_weight']]

        # training loops
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / self.run_configuration['batch_size'])
        for epoch in range(self.run_configuration['initial_epoch'], self.run_configuration['epochs']):

            model.train()

            # save model checkpoint
            if epoch != 0 and epoch % self.run_configuration['save_checkpoint'] == 0:
                model.save(os.path.join(
                    self.path_dir_run, f"model_{epoch:05d}.pt"))

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for step in range(steps_per_epoch):
                torch.cuda.empty_cache()

                step_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                inputs, y_true = next(generator)
                inputs = [d.to(device).float() for d in inputs]
                # inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
                y_true = [d.to(device).float() for d in y_true]
                # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

                # run inputs through the model to produce a warped image and flow field
                y_pred = model(*inputs)

                # calculate total loss
                loss = 0
                loss_list = []
                for n, loss_function in enumerate(losses):
                    curr_loss = loss_function(
                        y_true[n], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

            # Validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for i in range(len(self.val_dataset)):
                        val_inputs, val_y_true = next(val_generator)
                        val_inputs = [d.to(device).float() for d in val_inputs]
                        val_y_true = [d.to(device).float() for d in val_y_true]

                        val_y_pred = model(*val_inputs)

                        val_loss = 0

                        for n, loss_function in enumerate(losses):
                            val_curr_loss = loss_function(
                                val_y_true[n], val_y_pred[n]) * weights[n]
                            val_loss += val_curr_loss
                        val_loss_list.append(val_loss.item())

            # print epoch info
            epoch_info = 'Epoch %d/%d' % (epoch + 1,
                                          self.run_configuration['epochs'])
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            mean_loss = np.mean(epoch_loss, axis=0)
            losses_info = ', '.join(['%.4e' % f for f in mean_loss])
            if self.val_dataset is not None:
                loss_info = 'loss: %.4e  (%s), validation loss: %.4e' % (
                    np.mean(epoch_total_loss), losses_info, np.mean(val_loss_list))
            else:
                loss_info = 'loss: %.4e  (%s)' % (
                    np.mean(epoch_total_loss), losses_info)
            # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

            # wandb logging
            if self.use_wandb:
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(
                        epoch_total_loss), "sim-loss": mean_loss[0], "grad-loss": mean_loss[1], "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss),
                              "sim-loss": mean_loss[0], "grad-loss": mean_loss[1]})

        # final model save
        model.save(self.get_trained_model_path())
        self.model = model

        if self.use_wandb:
            wandb.finish()
