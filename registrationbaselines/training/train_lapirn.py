import registrationbaselines.core.utils_dl
import os
import time
import sys
from pathlib import Path

import wandb
import torch
from typing import Optional
from torch.utils.data import DataLoader
import numpy as np

# if we dont do this then LapIRN.Code.miccai2020_model_stage.py can't import Functions
sys.path.append(str(Path(__file__).parent.absolute().parent))
sys.path.append(
    str(Path(__file__).parent.parent.absolute() / "dl_repos/LapIRN/Code"))

from registrationbaselines.interfaces._interface_training import TrainingInterface  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.core import utils  # nopep8
from registrationbaselines.dl_repos.LapIRN.Code.Functions import generate_grid, transform_unit_flow_to_flow_cuda  # nopep8
from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC  # nopep8
from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_add_lvl2, Miccai2020_LDR_laplacian_unit_add_lvl3  # nopep8

import gc  # nopep8
gc.collect()
torch.cuda.empty_cache()


class LapIRN(TrainingInterface):
    """
    Training procedure for LapIRN network (https://github.com/cwmok/LapIRN)
    """

    def __init__(self,
                 train_dataset: data_loaders.GenericDataset,
                 configuration_path: Path,
                 val_dataset: Optional[data_loaders.GenericDataset] = None) -> None:
        """
        Initialization of LapIRN Training.
        @param train_dataset: training dataset
        @param config_path: path of config file
        @param val_dataset: validation dataset (optional)
        """

        super().__init__("LapIRN",
                         configuration_path,
                         train_dataset,
                         val_dataset)

        new_shape = registrationbaselines.core.utils_dl.get_new_lapirn_image_shape(
            self.train_dataset.image_shape)

        self.train_dataset.image_shape = tuple(new_shape)
        if self.val_dataset is not None:
            self.val_dataset.image_shape = tuple(new_shape)

        self.image_shape = self.train_dataset.image_shape
        self.image_shape2 = tuple(int(x / 2)
                                  for x in self.train_dataset.image_shape)
        self.image_shape4 = tuple(int(x / 4)
                                  for x in self.train_dataset.image_shape)

    def _train(self) -> None:
        """
        Training of the model.
        All three levels are trained sequentially.
        @return:
        """

        self._train_lvl1()
        self._train_lvl2()
        self._train_lvl3()

    def _train_lvl1(self) -> None:
        """
        Train level 1 with 1/4 image resolution
        @return:
        """
        print("Training lvl1...")

        if self.run_configuration["use_diff_version"]:
            model = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape4,
                                                           range_flow=self.run_configuration['range_flow']).to(self.device)
        else:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, self.run_configuration['start_channel'], is_train=True,
                                                                imgshape=self.image_shape4,
                                                                range_flow=self.run_configuration['range_flow']).to(self.device)

        loss_similarity = NCC(win=3)
        loss_Jdet = neg_Jdet_loss
        loss_smooth = smoothloss

        transform = SpatialTransform_unit().to(self.device)

        torch.save(model.state_dict(), os.path.join(
            self.path_dir_run, "model_initial.pt"))  # save initial model weights

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        grid_4 = generate_grid(self.image_shape4)
        grid_4 = torch.from_numpy(np.reshape(
            grid_4, (1,) + grid_4.shape)).to(self.device).float()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.run_configuration['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        lossall = np.zeros((4, self.run_configuration['iteration_lvl1'] + 1))

        # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
        #                                      shuffle=True, num_workers=2)

        training_generator = DataLoader(
            self.train_dataset, batch_size=self.run_configuration['batch_size'], shuffle=True)
        if self.val_dataset is not None:
            val_generator = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False)

        step = 0
        if self.run_configuration['load_model'] is not None:
            step = 3000
            model.load_state_dict(torch.load(
                self.base_dir / self.run_configuration['load_model']))
            # temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
            # lossall[:, 0:3000] = temp_lossall[:, 0:3000]

        while step <= self.run_configuration['iteration_lvl1']:

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for item in training_generator:
                X = item["moving_image"].unsqueeze(1)  # (bs,c,h,w,d)
                Y = item["fixed_image"].unsqueeze(1)  # (bs,c,h,w,d)
                X = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                    X, self.image_shape)
                Y = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                    Y, self.image_shape)

                step_start_time = time.time()

                X = X.to(self.device).float()
                Y = Y.to(self.device).float()

                # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                    F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                loss_regulation = loss_smooth(F_X_Y)

                loss = loss_multiNCC + self.run_configuration['antifold_weight'] * \
                    loss_Jacobian + \
                    self.run_configuration['smooth_weight'] * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                loss_list = [loss.item(), loss_multiNCC.item(),
                             loss_Jacobian.item(), loss_regulation.item()]
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                # print(
                #     "\r" + 'step {0}/{1} -> training loss {2:.4f} - sim_NCC {3:4f} - Jdet {4:.10f} -smo {5:.4f} -time {6}'.format(
                #         step, self.run_configuration['iteration_lvl1'], loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), epoch_step_time[-1]), flush=True)

                # with lr 1e-3 + with bias
                # if step > 0 and (step % self.run_configuration['save_checkpoint'] == 0):
                #     modelname =self.base_dir / self.run_configuration['model_path'] / (self.run_configuration['model_name'] + "stagelvl1_" + str(step) + '.pth')
                #     torch.save(model.state_dict(), modelname)
                #     # np.save(self.base_dir / self.run_configuration['model_path'] / ('loss' + self.run_configuration['model_name'] + "stagelvl1_" + str(step) + '.npy'), lossall)

                step += 1

                if step > self.run_configuration['iteration_lvl1']:
                    break

            # validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for item in val_generator:
                        X = item["moving_image"].unsqueeze(1)  # (bs,c,h,w,d)
                        Y = item["fixed_image"].unsqueeze(1)  # (bs,c,h,w,d)
                        X = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                            X, self.image_shape)
                        Y = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                            Y, self.image_shape)

                        X = X.to(self.device).float()
                        Y = Y.to(self.device).float()

                        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                        F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

                        # 3 level deep supervision NCC
                        loss_multiNCC = loss_similarity(X_Y, Y_4x)
                        F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                            F_X_Y.permute(0, 2, 3, 4, 1).clone())
                        loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

                        # reg2 - use velocity
                        _, _, x, y, z = F_X_Y.shape
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                        loss_regulation = loss_smooth(F_X_Y)

                        loss = loss_multiNCC + self.run_configuration['antifold_weight'] * loss_Jacobian + self.run_configuration[
                            'smooth_weight'] * loss_regulation

                        val_loss_list.append(loss.item())

            # wandb logging
            if self.use_wandb:
                mean_loss = np.mean(epoch_loss, axis=0)
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3],
                               "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss),
                              "sim-loss": mean_loss[1], "grad-loss": mean_loss[3]})

        torch.save(model.state_dict(), os.path.join(
            self.path_dir_run, "model_level1_final.pt"))

    def _train_lvl2(self):
        """
        Train level 2 with 1/2 image resolution
        @return:
        """
        print("Training lvl2...")

        if self.run_configuration["use_diff_version"]:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.run_configuration['start_channel'], is_train=True,
                                                                imgshape=self.image_shape4,
                                                                range_flow=self.run_configuration['range_flow']).to(self.device)
        else:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, self.run_configuration['start_channel'], is_train=True,
                                                                     imgshape=self.image_shape4,
                                                                     range_flow=self.run_configuration['range_flow']).to(self.device)

        model_lvl1.load_state_dict(torch.load(os.path.join(
            self.path_dir_run, "model_level1_final.pt")))

        # Freeze model_lvl1 weight
        for param in model_lvl1.parameters():
            param.requires_grad = False

        if self.run_configuration["use_diff_version"]:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape2,
                                                                range_flow=self.run_configuration['range_flow'], model_lvl1=model_lvl1).to(self.device)
        else:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, self.run_configuration['start_channel'], is_train=True,
                                                                imgshape=self.image_shape2,
                                                                range_flow=self.run_configuration['range_flow'],
                                                                model_lvl1=model_lvl1).to(self.device)

        loss_similarity = multi_resolution_NCC(win=5, scale=2)
        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(self.device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        grid_2 = generate_grid(self.image_shape2)
        grid_2 = torch.from_numpy(np.reshape(
            grid_2, (1,) + grid_2.shape)).to(self.device).float()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.run_configuration['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        lossall = np.zeros((4, self.run_configuration['iteration_lvl2'] + 1))

        # training_generator = DataLoader(Dataset_epoch(names, norm=False), batch_size=self.run_configuration['batch_size'],
        #                                      shuffle=True, num_workers=2)
        training_generator = DataLoader(
            self.train_dataset, batch_size=self.run_configuration['batch_size'], shuffle=True)
        if self.val_dataset is not None:
            val_generator = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False)

        step = 0
        if self.run_configuration['load_model'] is not None:
            step = 3000
            model.load_state_dict(torch.load(
                self.base_dir / self.run_configuration['load_model']))

        while step <= self.run_configuration['iteration_lvl2']:

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for item in training_generator:
                X = item["moving_image"].unsqueeze(1)  # (bs,c,h,w,d)
                Y = item["fixed_image"].unsqueeze(1)  # (bs,c,h,w,d)
                X = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                    X, self.image_shape)
                Y = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                    Y, self.image_shape)

                step_start_time = time.time()

                X = X.to(self.device).float()
                Y = Y.to(self.device).float()

                # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                    F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                loss_regulation = loss_smooth(F_X_Y)

                loss = loss_multiNCC + self.run_configuration['antifold_weight'] * \
                    loss_Jacobian + \
                    self.run_configuration['smooth_weight'] * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                loss_list = [loss.item(), loss_multiNCC.item(),
                             loss_Jacobian.item(), loss_regulation.item()]
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                # print(
                #     "\r" + 'step {0}/{1} -> training loss {2:.4f} - sim_NCC {3:4f} - Jdet {4:.10f} -smo {5:.4f} -time {6}'.format(
                #         step, self.run_configuration['iteration_lvl2'], loss.item(
                #         ), loss_multiNCC.item(), loss_Jacobian.item(),
                #         loss_regulation.item(), epoch_step_time[-1]), flush=True)

                # with lr 1e-3 + with bias
                # if (step % self.run_configuration['save_checkpoint'] == 0):
                #     modelname = self.base_dir / self.run_configuration['model_path'] / (self.run_configuration['model_name'] + 'stagelvl2_' + str(step) + '.pth')
                #     torch.save(model.state_dict(), modelname)
                #     # np.save(self.base_dir / self.run_configuration['model_path'] / ('loss' + self.run_configuration['model_name'] + 'stagelvl2_' + str(step) + '.npy'), lossall)

                if step == self.run_configuration['freeze_step']:
                    model.unfreeze_modellvl1()

                step += 1

                if step > self.run_configuration['iteration_lvl2']:
                    break

            # validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for item in val_generator:
                        X = item["moving_image"].unsqueeze(1)  # (bs,c,h,w,d)
                        Y = item["fixed_image"].unsqueeze(1)  # (bs,c,h,w,d)
                        X = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                            X, self.image_shape)
                        Y = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                            Y, self.image_shape)
                        X = X.to(self.device).float()
                        Y = Y.to(self.device).float()

                        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

                        # 3 level deep supervision NCC
                        loss_multiNCC = loss_similarity(X_Y, Y_4x)

                        F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                            F_X_Y.permute(0, 2, 3, 4, 1).clone())

                        loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

                        # reg2 - use velocity
                        _, _, x, y, z = F_X_Y.shape
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                        loss_regulation = loss_smooth(F_X_Y)

                        loss = loss_multiNCC + self.run_configuration['antifold_weight'] * loss_Jacobian + self.run_configuration[
                            'smooth_weight'] * loss_regulation

                        val_loss_list.append(loss.item())

            # wandb logging
            if self.use_wandb:
                mean_loss = np.mean(epoch_loss, axis=0)
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3],
                               "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss),
                              "sim-loss": mean_loss[1], "grad-loss": mean_loss[3]})

        # modelname = self.base_dir / self.run_configuration['model_path'] / (self.run_configuration['model_name'] + 'stagelvl2_final.pth')
        torch.save(model.state_dict(), os.path.join(
            self.path_dir_run, "model_level2_final.pt"))
        # np.save(self.base_dir / self.run_configuration['model_path'] / ('loss' + self.run_configuration['model_name'] + 'stagelvl2.npy'),
        # lossall)

    def _train_lvl3(self):
        """
        Train level 3 with full image resolution
        @return:
        """
        print("Training lvl3...")

        if self.run_configuration["use_diff_version"]:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape4,
                                                                range_flow=self.run_configuration['range_flow']).to(self.device)
            model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape2,
                                                                range_flow=self.run_configuration['range_flow'], model_lvl1=model_lvl1).to(self.device)
        else:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape4,
                                                                     range_flow=self.run_configuration['range_flow']).to(self.device)
            model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape2,
                                                                     range_flow=self.run_configuration['range_flow'], model_lvl1=model_lvl1).to(self.device)

        model_lvl2.load_state_dict(torch.load(os.path.join(
            self.path_dir_run, "model_level2_final.pt")))

        # Freeze model_lvl1 weight
        for param in model_lvl2.parameters():
            param.requires_grad = False

        if self.run_configuration["use_diff_version"]:
            model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, self.run_configuration['start_channel'], is_train=True, imgshape=self.image_shape,
                                                           range_flow=self.run_configuration['range_flow'], model_lvl2=model_lvl2).to(self.device)
        else:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, self.run_configuration['start_channel'], is_train=True,
                                                                imgshape=self.image_shape,
                                                                range_flow=self.run_configuration['range_flow'],
                                                                model_lvl2=model_lvl2).to(self.device)

        loss_similarity = multi_resolution_NCC(win=7, scale=3)
        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(self.device)
        # transform_nearest = SpatialTransformNearest_unit().to(self.device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        grid = generate_grid(self.image_shape)
        grid = torch.from_numpy(np.reshape(
            grid, (1,) + grid.shape)).to(self.device).float()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.run_configuration['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        lossall = np.zeros((4, self.run_configuration['iteration_lvl3'] + 1))

        # training_generator = DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
        #                                      shuffle=True, num_workers=2)
        training_generator = DataLoader(
            self.train_dataset, batch_size=self.run_configuration['batch_size'], shuffle=True)
        if self.val_dataset is not None:
            val_generator = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False)

        step = 0
        if self.run_configuration['load_model'] is not None:
            step = 3000
            model.load_state_dict(torch.load(
                self.base_dir / self.run_configuration['load_model']))

        while step <= self.run_configuration['iteration_lvl3']:

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for item in training_generator:
                X = item["moving_image"].unsqueeze(1)  # (bs,c,h,w,d)
                Y = item["fixed_image"].unsqueeze(1)  # (bs,c,h,w,d)
                X = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                    X, self.image_shape)
                Y = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                    Y, self.image_shape)
                X = X.to(self.device).float()
                Y = Y.to(self.device).float()

                step_start_time = time.time()

                # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                    F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                loss_regulation = loss_smooth(F_X_Y)

                loss = loss_multiNCC + self.run_configuration['antifold_weight'] * \
                    loss_Jacobian + \
                    self.run_configuration['smooth_weight'] * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                loss_list = [loss.item(), loss_multiNCC.item(),
                             loss_Jacobian.item(), loss_regulation.item()]
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                lossall[:, step] = np.array(
                    loss_list)
                # print(
                #     "\r" + 'step {0}/{1} -> training loss {2:.4f} - sim_NCC {3:4f} - Jdet {4:.10f} -smo {5:.4f} -time {6}'.format(
                #         step, self.run_configuration['iteration_lvl3'], loss.item(), loss_multiNCC.item(), loss_Jacobian.item(),
                #         loss_regulation.item(), epoch_step_time[-1]), flush=True)

                # with lr 1e-3 + with bias
                # if (step % self.run_configuration['save_checkpoint'] == 0):
                #     modelname = self.base_dir / self.run_configuration['model_path'] / (self.run_configuration['model_name'] + 'stagelvl3_' + str(step) + '.pth')
                #     torch.save(model.state_dict(), modelname)
                #     #np.save(self.base_dir / self.run_configuration['model_path'] / ('loss' + self.run_configuration['model_name'] + 'stagelvl3_' + str(step) + '.npy'), lossall)

                # Validation

                if step == self.run_configuration['freeze_step']:
                    model.unfreeze_modellvl2()

                step += 1

                if step > self.run_configuration['iteration_lvl3']:
                    break

            # validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for item in val_generator:
                        X = item["moving_image"].unsqueeze(1)  # (bs,c,h,w,d)
                        Y = item["fixed_image"].unsqueeze(1)  # (bs,c,h,w,d)
                        X = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                            X, self.image_shape)
                        Y = registrationbaselines.core.utils_dl.pad_tensor_to_shape(
                            Y, self.image_shape)
                        X = X.to(self.device).float()
                        Y = Y.to(self.device).float()

                        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(
                            X, Y)

                        # 3 level deep supervision NCC
                        loss_multiNCC = loss_similarity(X_Y, Y_4x)

                        F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                            F_X_Y.permute(0, 2, 3, 4, 1).clone())

                        loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                        # reg2 - use velocity
                        _, _, x, y, z = F_X_Y.shape
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                        loss_regulation = loss_smooth(F_X_Y)

                        loss = loss_multiNCC + self.run_configuration['antifold_weight'] * loss_Jacobian + self.run_configuration[
                            'smooth_weight'] * loss_regulation

                        val_loss_list.append(loss.item())

            # wandb logging
            if self.use_wandb:
                mean_loss = np.mean(epoch_loss, axis=0)
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(
                        epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3], "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss),
                              "sim-loss": mean_loss[1], "grad-loss": mean_loss[3]})

        # modelname = self.base_dir / self.run_configuration['model_path'] / (self.run_configuration['model_name'] + 'stagelvl3_final.pth')
        torch.save(model.state_dict(), os.path.join(
            self.path_dir_run, "model_level3_final.pt"))
        # np.save(self.base_dir / self.run_configuration['model_path'] / ('loss' + self.run_configuration['model_name'] + 'stagelvl3.npy'), lossall)
