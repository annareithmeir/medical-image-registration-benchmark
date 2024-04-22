import time
from pathlib import Path
import os
import wandb
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))

# if we dont do this then LapIRN.Code.miccai2020_model_stage.py can't import Functions
sys.path.append(str(Path(__file__).parent.parent.absolute() / "dl_repos/LapIRN/Code"))
print(sys.path)

from registrationbaselines.core.training_interface import TrainingInterface

from registrationbaselines.dl_repos.LapIRN.Code.Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC
from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_add_lvl2, Miccai2020_LDR_laplacian_unit_add_lvl3

import gc
gc.collect()
torch.cuda.empty_cache()


class LapIRNTraining(TrainingInterface):
    """
    Training procedure for LapIRN by T. Mok (https://github.com/cwmok/LapIRN)
    """

    def __init__(self, train_dataset: Dataset, config_path: Path(), val_dataset: Dataset=None):
        self.method = "lapirn"

        # paths
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = self.read_config(config_path)
        self.base_dir = Path(__file__).parent.parent.absolute().parent

        print(self.config)

        if self.config['use_wandb']:
            self.init_wandb(self.base_dir / self.config['wandb_config_path'])

        self.imgshape = self.train_dataset.img_shape
        self.imgshape2 = tuple(int(x / 2) for x in self.train_dataset.img_shape)
        self.imgshape4 = tuple(int(x / 4) for x in self.train_dataset.img_shape)
        print(self.imgshape, self.imgshape2, self.imgshape4)

        self.use_diff_version = self.config["use_diff_version"]

        if not os.path.isdir(self.base_dir / self.config['result_model_path']):
            os.mkdir(self.base_dir / self.config['result_model_path'])

    def train(self):
        self.train_lvl1()
        self.train_lvl2()
        self.train_lvl3()

    def train_lvl1(self):
        print("Training lvl1...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        if self.use_diff_version:
            model = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape4,
                                                                range_flow=self.config['range_flow']).to(device)
        else:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, self.config['start_channel'], is_train=True,
                                                                imgshape=self.imgshape4,
                                                                range_flow=self.config['range_flow']).to(device)

        loss_similarity = NCC(win=3)
        loss_Jdet = neg_Jdet_loss
        loss_smooth = smoothloss

        transform = SpatialTransform_unit().to(device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        grid_4 = generate_grid(self.imgshape4)
        grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        lossall = np.zeros((4, self.config['iteration_lvl1'] + 1))

        # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
        #                                      shuffle=True, num_workers=2)

        training_generator = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        if self.val_dataset is not None:
            val_generator = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        step = 0
        if self.config['load_model'] is not None:
            print("Loading model from: ", self.base_dir / self.config['load_model'])
            step = 3000
            model.load_state_dict(torch.load(self.base_dir / self.config['load_model']))
            #temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
            #lossall[:, 0:3000] = temp_lossall[:, 0:3000]

        while step <= self.config['iteration_lvl1']:

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for X, Y in training_generator:

                step_start_time = time.time()

                X = X.to(device).float()
                Y = Y.to(device).float()

                # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                loss_regulation = loss_smooth(F_X_Y)

                loss = loss_multiNCC + self.config['antifold_weight'] * loss_Jacobian + self.config['smooth_weight'] * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                loss_list = [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()]
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                print(
                    "\r" + 'step {0}/{1} -> training loss {2:.4f} - sim_NCC {3:4f} - Jdet {4:.10f} -smo {5:.4f} -time {6}'.format(
                        step, self.config['iteration_lvl1'], loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), epoch_step_time[-1]), flush=True)

                # with lr 1e-3 + with bias
                if (step % self.config['save_checkpoint'] == 0):
                    modelname =self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + "stagelvl1_" + str(step) + '.pth')
                    torch.save(model.state_dict(), modelname)
                    np.save(self.base_dir / self.config['result_model_path'] / ('loss' + self.config['model_name'] + "stagelvl1_" + str(step) + '.npy'), lossall)

                step += 1

                if step > self.config['iteration_lvl1']:
                    break
            print("one epoch pass")

            #validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for X, Y in val_generator:

                        X = X.to(device).float()
                        Y = Y.to(device).float()

                        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                        F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

                        # 3 level deep supervision NCC
                        loss_multiNCC = loss_similarity(X_Y, Y_4x)
                        F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())
                        loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

                        # reg2 - use velocity
                        _, _, x, y, z = F_X_Y.shape
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                        loss_regulation = loss_smooth(F_X_Y)

                        loss = loss_multiNCC + self.config['antifold_weight'] * loss_Jacobian + self.config[
                            'smooth_weight'] * loss_regulation

                        val_loss_list.append(loss.item())

            # wandb logging
            if self.config['use_wandb']:
                mean_loss = np.mean(epoch_loss, axis=0)
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3],
                               "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3]})

        modelname = self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl1_final.pth')
        torch.save(model.state_dict(), modelname)
        np.save(self.base_dir / self.config['result_model_path'] / ('loss' + self.config['model_name'] + 'stagelvl1.npy'), lossall)

    def train_lvl2(self):
        print("Training lvl2...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.use_diff_version:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.config['start_channel'], is_train=True,
                                                                     imgshape=self.imgshape4,
                                                                     range_flow=self.config['range_flow']).to(device)
        else:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, self.config['start_channel'], is_train=True,
                                                                     imgshape=self.imgshape4,
                                                                     range_flow=self.config['range_flow']).to(device)


        model_lvl1.load_state_dict(torch.load(self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl1_final.pth')))
        print("Loading weight for model_lvl1...", self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl1_final.pth'))

        # Freeze model_lvl1 weight
        for param in model_lvl1.parameters():
            param.requires_grad = False

        if self.use_diff_version:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape2,
                                                                range_flow=self.config['range_flow'], model_lvl1=model_lvl1).to(device)
        else:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, self.config['start_channel'], is_train=True,
                                                                imgshape=self.imgshape2,
                                                                range_flow=self.config['range_flow'],
                                                                model_lvl1=model_lvl1).to(device)

        loss_similarity = multi_resolution_NCC(win=5, scale=2)
        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        grid_2 = generate_grid(self.imgshape2)
        grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        lossall = np.zeros((4, self.config['iteration_lvl2'] + 1))

        # training_generator = DataLoader(Dataset_epoch(names, norm=False), batch_size=self.config['batch_size'],
        #                                      shuffle=True, num_workers=2)
        training_generator = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        if self.val_dataset is not None:
            val_generator = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        step = 0
        if self.config['load_model'] is not None:
            print("Loading model from: ", self.base_dir / self.config['load_model'])
            step = 3000
            model.load_state_dict(torch.load(self.base_dir / self.config['load_model']))

        while step <= self.config['iteration_lvl2']:

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for X, Y in training_generator:

                step_start_time = time.time()

                X = X.to(device).float()
                Y = Y.to(device).float()

                # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                loss_regulation = loss_smooth(F_X_Y)

                loss = loss_multiNCC + self.config['antifold_weight'] * loss_Jacobian + self.config['smooth_weight'] * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                loss_list = [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()]
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                print(
                    "\r" + 'step {0}/{1} -> training loss {2:.4f} - sim_NCC {3:4f} - Jdet {4:.10f} -smo {5:.4f} -time {6}'.format(
                        step, self.config['iteration_lvl2'], loss.item(), loss_multiNCC.item(), loss_Jacobian.item(),
                        loss_regulation.item(), epoch_step_time[-1]), flush=True)

                # with lr 1e-3 + with bias
                if (step % self.config['save_checkpoint'] == 0):
                    modelname = self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl2_' + str(step) + '.pth')
                    torch.save(model.state_dict(), modelname)
                    np.save(self.base_dir / self.config['result_model_path'] / ('loss' + self.config['model_name'] + 'stagelvl2_' + str(step) + '.npy'), lossall)

                if step == self.config['freeze_step']:
                    model.unfreeze_modellvl1()

                step += 1

                if step > self.config['iteration_lvl2']:
                    break
            print("one epoch pass")

            # validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for X, Y in val_generator:
                        X = X.to(device).float()
                        Y = Y.to(device).float()

                        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

                        # 3 level deep supervision NCC
                        loss_multiNCC = loss_similarity(X_Y, Y_4x)

                        F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

                        loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

                        # reg2 - use velocity
                        _, _, x, y, z = F_X_Y.shape
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                        loss_regulation = loss_smooth(F_X_Y)

                        loss = loss_multiNCC + self.config['antifold_weight'] * loss_Jacobian + self.config[
                            'smooth_weight'] * loss_regulation

                        val_loss_list.append(loss.item())


            # wandb logging
            if self.config['use_wandb']:
                mean_loss = np.mean(epoch_loss, axis=0)
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3],
                               "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3]})

        modelname = self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl2_final.pth')
        torch.save(model.state_dict(), modelname)
        np.save(self.base_dir / self.config['result_model_path'] / ('loss' + self.config['model_name'] + 'stagelvl2.npy'),
                lossall)

    def train_lvl3(self):
        print("Training lvl3...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.use_diff_version:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape4,
                                                                     range_flow=self.config['range_flow']).to(device)
            model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape2,
                                                                     range_flow=self.config['range_flow'], model_lvl1=model_lvl1).to(device)
        else:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape4,
                                                                     range_flow=self.config['range_flow']).to(device)
            model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape2,
                                                                     range_flow=self.config['range_flow'], model_lvl1=model_lvl1).to(device)

        model_lvl2.load_state_dict(torch.load(self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl2_final.pth')))
        print("Loading weight for model_lvl2...", self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl2_final.pth'))

        # Freeze model_lvl1 weight
        for param in model_lvl2.parameters():
            param.requires_grad = False

        if self.use_diff_version:
            model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, self.config['start_channel'], is_train=True, imgshape=self.imgshape,
                                                                range_flow=self.config['range_flow'], model_lvl2=model_lvl2).to(device)
        else:
            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, self.config['start_channel'], is_train=True,
                                                                imgshape=self.imgshape,
                                                                range_flow=self.config['range_flow'],
                                                                model_lvl2=model_lvl2).to(device)

        loss_similarity = multi_resolution_NCC(win=7, scale=3)
        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(device)
        # transform_nearest = SpatialTransformNearest_unit().to(device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        grid = generate_grid(self.imgshape)
        grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        lossall = np.zeros((4, self.config['iteration_lvl3'] + 1))

        # training_generator = DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
        #                                      shuffle=True, num_workers=2)
        training_generator = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        if self.val_dataset is not None:
            val_generator = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        step = 0
        if self.config['load_model'] is not None:
            print("Loading model from: ", self.base_dir / self.config['load_model'])
            step = 3000
            model.load_state_dict(torch.load(self.base_dir / self.config['load_model']))

        while step <= self.config['iteration_lvl3']:

            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []

            for X, Y in training_generator:

                step_start_time = time.time()

                X = X.to(device).float()
                Y = Y.to(device).float()

                # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                loss_regulation = loss_smooth(F_X_Y)

                loss = loss_multiNCC + self.config['antifold_weight'] * loss_Jacobian + self.config['smooth_weight'] * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                loss_list = [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()]
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                lossall[:, step] = np.array(
                    loss_list)
                print(
                    "\r" + 'step {0}/{1} -> training loss {2:.4f} - sim_NCC {3:4f} - Jdet {4:.10f} -smo {5:.4f} -time {6}'.format(
                        step, self.config['iteration_lvl3'], loss.item(), loss_multiNCC.item(), loss_Jacobian.item(),
                        loss_regulation.item(), epoch_step_time[-1]), flush=True)

                # with lr 1e-3 + with bias
                if (step % self.config['save_checkpoint'] == 0):
                    modelname = self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl3_' + str(step) + '.pth')
                    torch.save(model.state_dict(), modelname)
                    np.save(self.base_dir / self.config['result_model_path'] / ('loss' + self.config['model_name'] + 'stagelvl3_' + str(step) + '.npy'), lossall)

                    # Validation

                if step == self.config['freeze_step']:
                    model.unfreeze_modellvl2()

                step += 1

                if step > self.config['iteration_lvl3']:
                    break
            print("one epoch pass")

            # validation
            val_loss_list = list()
            if self.val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    for X, Y in val_generator:
                        X = X.to(device).float()
                        Y = Y.to(device).float()

                        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

                        # 3 level deep supervision NCC
                        loss_multiNCC = loss_similarity(X_Y, Y_4x)

                        F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

                        loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                        # reg2 - use velocity
                        _, _, x, y, z = F_X_Y.shape
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z - 1)
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x - 1)
                        loss_regulation = loss_smooth(F_X_Y)

                        loss = loss_multiNCC + self.config['antifold_weight'] * loss_Jacobian + self.config[
                            'smooth_weight'] * loss_regulation

                        val_loss_list.append(loss.item())

            # wandb logging
            if self.config['use_wandb']:
                mean_loss = np.mean(epoch_loss, axis=0)
                if self.val_dataset is not None:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3], "val-loss": np.mean(val_loss_list)})
                else:
                    wandb.log({"loss": np.mean(epoch_total_loss), "sim-loss": mean_loss[1], "grad-loss": mean_loss[3]})

        modelname = self.base_dir / self.config['result_model_path'] / (self.config['model_name'] + 'stagelvl3_final.pth')
        torch.save(model.state_dict(), modelname)
        np.save(self.base_dir / self.config['result_model_path'] / ('loss' + self.config['model_name'] + 'stagelvl3.npy'), lossall)

    def get_trained_model_path(self):
        return self.config['result_model_path']

    def get_initial_weights_path(self):
        return self.initial_weights_path

    def save_initial_weights(self):
        assert self.model is not None, "Model is not yet initialized!"
        torch.save(self.model.state_dict(), self.base_dir / self.config['initial_weights_path']) # '.pth'

    def init_wandb(self, wandb_config_path):
        wandb_config = self.read_config(wandb_config_path)
        wandb.init(
            project=wandb_config['project'],
            group=wandb_config['group'],
            name=wandb_config['name'],
            config=wandb_config['config_dict']
        )


