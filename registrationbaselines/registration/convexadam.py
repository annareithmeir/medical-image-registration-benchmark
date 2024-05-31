import time
from pathlib import Path
from typing import Optional, Union
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent.parent))

import registrationbaselines.core.utils as utils
from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.dl_repos.convexAdam.src.convexAdam.convex_adam_utils import MINDSSC, correlate, coupled_convex, inverse_consistency
from registrationbaselines.core.utils_nifti import set_intent_code

import gc
gc.collect()
torch.cuda.empty_cache()


class ConvexAdam(RegistrationInterface):
    """
    Interface for the ConvexAdam registration method (https://github.com/multimodallearning/ConvexAdam) by Hansen, Heinrich 2021
    """
    def __init__(self, config_path: Path):
        self.method = "ConvexAdam"
        self.base_dir = Path(__file__).parent.parent.absolute().parent
        self.configuration = self.read_config(config_path)

        self._create_result_directories()

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_transformation_path = Path()

        self.feature_type = self.configuration["features"]

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False) -> None:
        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path

        # check that both images exist
        assert self.fixed_path.exists(), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(), f"File {self.moving_path} does not exist."

        image_moving = torch.from_numpy(utils.load_image_from_nii_gz(self.moving_path))
        image_fixed = torch.from_numpy(utils.load_image_from_nii_gz(self.fixed_path))
        assert image_fixed.shape == image_moving.shape

        self.result_transformed_image_path, \
            self.result_transformation_path = self._create_result_paths(self.fixed_path.stem,
                                                                      self.moving_path.stem,
                                                                      ".nii.gz",
                                                                      ".nii.gz")

        if self.feature_type == "MIND":
            displacement_field = self.convex_adam_mind(
                image_fixed=image_fixed,
                image_moving=image_moving,
                mind_r=self.configuration['mind_r'],
                mind_d=self.configuration['mind_d'],
                lambda_weight=self.configuration['lambda_weight'],
                grid_sp=self.configuration['grid_sp'],
                disp_hw=self.configuration['disp_hw'],
                selected_niter=self.configuration['selected_niter'],
                selected_smooth=self.configuration['selected_smooth'],
                grid_sp_adam=self.configuration['grid_sp_adam'],
                ic=self.configuration['ic'],
                use_mask=self.configuration['use_mask'],
                path_fixed_mask=None,
                path_moving_mask=None
            )

        elif self.feature_type=="nnunet":
            displacement_field = self.convex_adam_nnunet(
                image_fixed=image_fixed,
                image_moving=image_moving,
                lambda_weight=self.configuration['lambda_weight'],
                grid_sp=self.configuration['grid_sp'],
                disp_hw=self.configuration['disp_hw'],
                selected_niter=self.configuration['selected_niter'],
                selected_smooth=self.configuration['selected_smooth'],
                grid_sp_adam=self.configuration['grid_sp_adam'],
                ic=self.configuration['ic']
            )
        else:
            print("Feature type must be either MIND or nnunet! Wrong type given in config file.")

        H, W, D = image_moving.shape

        displacement_field = np.expand_dims(displacement_field, axis=3)
        image_warped = F.grid_sample(image_moving.float().view(1,1,H,W,D),torch.from_numpy(displacement_field).float().view(1,H,W,D,3),align_corners=False,mode='nearest').numpy().squeeze()
        self._save_results(image_warped, displacement_field.squeeze())
        set_intent_code(self.result_transformation_path, 'NIFTI_INTENT_DISPVECT')

    def convex_adam_nnunet(self,
                           image_fixed: Union[torch.Tensor, np.ndarray, sitk.Image],
                           image_moving: Union[torch.Tensor, np.ndarray, sitk.Image],
                           lambda_weight: float = 1.25,
                           grid_sp: int = 6,
                           disp_hw: int = 4,
                           selected_niter: int = 80,
                           selected_smooth: int = 0,
                           grid_sp_adam: int = 2,
                           ic: bool = True) -> np.ndarray:
        """
        Inference for given image pair with nnUNet features
        @param image_fixed:
        @param image_moving:
        @param lambda_weight:
        @param grid_sp:
        @param disp_hw:
        @param selected_niter:
        @param selected_smooth:
        @param grid_sp_adam:
        @param ic:
        @return: displacement field as np.ndarray
        """

        image_fixed = self._validate_image(image_fixed)
        image_moving = self._validate_image(image_moving)
        image_fixed = image_fixed.float()
        image_moving = image_moving.float()

        H, W, D = image_fixed.shape[-3:]

        torch.cuda.synchronize()
        t0 = time.time()

        # compute features and downsample (using average pooling)
        with torch.no_grad():

            # todo get_nnunet_features(moving, fixed)

            features_fix, features_mov = self._extract_features_nnunet(image_fixed=image_fixed, image_moving=image_moving)

            features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
            features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)

            n_ch = features_fix_smooth.shape[1]

        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
                                    align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)

        else:
            disp_hr = disp_soft

        # run Adam instance optimisation
        if lambda_weight > 0:
            with torch.no_grad():

                patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
                patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

            # create optimisable displacement grid
            disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                    mode='trilinear', align_corners=False)

            net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
            net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
            net.cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=1)

            grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                                  (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

            # run Adam optimisation with diffusion regularisation and B-spline smoothing
            for iter in range(selected_niter):
                optimizer.zero_grad()

                disp_sample = F.avg_pool3d(
                    F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3,
                    stride=1, padding=1).permute(0, 2, 3, 4, 1)
                reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                           lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                           lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

                scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                      (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
                grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

                patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                                  grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam,
                                                                 D // grid_sp_adam, 3).cuda(), align_corners=False,
                                                  mode='bilinear')

                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
                loss = sampled_cost.mean()
                (loss + reg_loss).backward()
                optimizer.step()

            fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
            disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

            if selected_smooth == 5:
                kernel_smooth = 5
                padding_smooth = kernel_smooth // 2
                disp_hr = F.avg_pool3d(
                    F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                                 padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

            if selected_smooth == 3:
                kernel_smooth = 3
                padding_smooth = kernel_smooth // 2
                disp_hr = F.avg_pool3d(
                    F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                                 padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        torch.cuda.synchronize()
        t1 = time.time()
        case_time = t1 - t0
        print('case time: ', case_time)

        x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
        y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
        z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
        displacement_field = np.stack((x, y, z), 3).astype(float)

        return displacement_field


    def convex_adam_mind(self,
            image_fixed: Union[torch.Tensor, np.ndarray, sitk.Image],
            image_moving: Union[torch.Tensor, np.ndarray, sitk.Image],
            mind_r: int = 1,
            mind_d: int = 2,
            lambda_weight: float = 1.25,
            grid_sp: int = 6,
            disp_hw: int = 4,
            selected_niter: int = 80,
            selected_smooth: int = 0,
            grid_sp_adam: int = 2,
            ic: bool = True,
            use_mask: bool = False,
            path_fixed_mask: Optional[Union[Path, str]] = None,
            path_moving_mask: Optional[Union[Path, str]] = None,
    ) -> np.ndarray:
        """
        Inference for given data with MIND features
        @param image_fixed:
        @param image_moving:
        @param mind_r:
        @param mind_d:
        @param lambda_weight:
        @param grid_sp:
        @param disp_hw:
        @param selected_niter:
        @param selected_smooth:
        @param grid_sp_adam:
        @param ic:
        @param use_mask:
        @param path_fixed_mask:
        @param path_moving_mask:
        @return: displacement field as np.ndarray
        """
        """Coupled convex optimisation with adam instance optimisation"""
        image_fixed = self._validate_image(image_fixed)
        image_moving = self._validate_image(image_moving)
        image_fixed = image_fixed.float()
        image_moving = image_moving.float()

        if use_mask:
            mask_fixed = torch.from_numpy(nib.load(path_fixed_mask).get_fdata()).float()
            mask_moving = torch.from_numpy(nib.load(path_moving_mask).get_fdata()).float()
        else:
            mask_fixed = None
            mask_moving = None

        H, W, D = image_fixed.shape

        torch.cuda.synchronize()
        t0 = time.time()

        # compute features and downsample (using average pooling)
        with torch.no_grad():

            features_fix, features_mov = self._extract_features_mind(image_fixed=image_fixed,
                                                          image_moving=image_moving,
                                                          mind_r=mind_r,
                                                          mind_d=mind_d,
                                                          use_mask=use_mask,
                                                          mask_fixed=mask_fixed,
                                                          mask_moving=mask_moving)

            features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
            features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)

            n_ch = features_fix_smooth.shape[1]

        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
                                    align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)

        else:
            disp_hr = disp_soft

        # run Adam instance optimisation
        if lambda_weight > 0:
            with torch.no_grad():
                patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
                patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

            # create optimisable displacement grid
            disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                    mode='trilinear', align_corners=False)

            net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
            net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
            net.cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=1)

            grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                                  (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

            # run Adam optimisation with diffusion regularisation and B-spline smoothing
            for iter in range(selected_niter):
                optimizer.zero_grad()

                disp_sample = F.avg_pool3d(
                    F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3,
                    stride=1, padding=1).permute(0, 2, 3, 4, 1)
                reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                           lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                           lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

                scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                      (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
                grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

                patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                                  grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam,
                                                                 D // grid_sp_adam, 3).cuda(), align_corners=False,
                                                  mode='bilinear')

                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
                loss = sampled_cost.mean()
                (loss + reg_loss).backward()
                optimizer.step()

            fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
            disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

            if selected_smooth == 5:
                kernel_smooth = 5
                padding_smooth = kernel_smooth // 2
                disp_hr = F.avg_pool3d(
                    F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                                 padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

            if selected_smooth == 3:
                kernel_smooth = 3
                padding_smooth = kernel_smooth // 2
                disp_hr = F.avg_pool3d(
                    F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                                 padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        torch.cuda.synchronize()
        t1 = time.time()
        case_time = t1 - t0
        print('case time: ', case_time)

        x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
        y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
        z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
        displacement_field = np.stack((x, y, z), 3).astype(float)

        return displacement_field


    def _save_results(self,  image_warped: np.ndarray, displacement_field: np.ndarray) -> None:
        """
        Saves the results to files
        @param image_warped: The warped image
        @param displacement_field: The displacement fie
        @return:
        """

        self.path_result_transformed_image, self.path_result_transformation = \
            self._create_result_paths(self.fixed_path.stem,
                                      self.moving_path.stem,
                                      ".nii.gz",
                                      ".nii.gz")

        affine = nib.load(self.fixed_path).affine
        print(self.result_transformation_path)

        utils.save_array_to_nii_gz_image(image_warped, self.result_transformed_image_path, affine=affine)
        utils.save_array_to_nii_gz_displacement_field(displacement_field, self.result_transformation_path, affine=affine)


    def _extract_features_mind(self,
            image_fixed: torch.Tensor,
            image_moving: torch.Tensor,
            mind_r: int,
            mind_d: int,
            use_mask: bool,
            mask_fixed: torch.Tensor,
            mask_moving: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """Extract MIND and/or semantic nnUNet features"""

        # MIND features
        if use_mask:
            H, W, D = image_fixed.shape[-3:]

            # replicate masking
            avg3 = nn.Sequential(nn.ReplicationPad3d(1), nn.AvgPool3d(3, stride=1))
            avg3.cuda()

            mask = (avg3(mask_fixed.view(1, 1, H, W, D).cuda()) > 0.9).float()
            _, idx = edt((mask[0, 0, ::2, ::2, ::2] == 0).squeeze().cpu().numpy(), return_indices=True)
            fixed_r = F.interpolate((image_fixed[::2, ::2, ::2].cuda().reshape(-1)[
                idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]]).unsqueeze(0).unsqueeze(0), scale_factor=2,
                                    mode='trilinear')
            fixed_r.view(-1)[mask.view(-1) != 0] = image_fixed.cuda().reshape(-1)[mask.view(-1) != 0]

            mask = (avg3(mask_moving.view(1, 1, H, W, D).cuda()) > 0.9).float()
            _, idx = edt((mask[0, 0, ::2, ::2, ::2] == 0).squeeze().cpu().numpy(), return_indices=True)
            moving_r = F.interpolate((image_moving[::2, ::2, ::2].cuda().reshape(-1)[
                idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]]).unsqueeze(0).unsqueeze(0), scale_factor=2,
                                     mode='trilinear')
            moving_r.view(-1)[mask.view(-1) != 0] = image_moving.cuda().reshape(-1)[mask.view(-1) != 0]

            features_fix = MINDSSC(fixed_r.cuda(), mind_r, mind_d).half()
            features_mov = MINDSSC(moving_r.cuda(), mind_r, mind_d).half()
        else:
            image_fixed = image_fixed.unsqueeze(0).unsqueeze(0)
            image_moving = image_moving.unsqueeze(0).unsqueeze(0)
            features_fix = MINDSSC(image_fixed.cuda(), mind_r, mind_d).half()
            features_mov = MINDSSC(image_moving.cuda(), mind_r, mind_d).half()

        return features_fix, features_mov

    def _extract_features_nnunet(self, image_fixed: torch.Tensor, image_moving: torch.Tensor,):
        # process nnUNet features

        # eps = 1e-32
        # H, W, D = pred_fixed.shape[-3:]
        #
        # combined_bins = torch.bincount(pred_fixed.long().reshape(-1)) + torch.bincount(pred_moving.long().reshape(-1))
        #
        # pos = torch.nonzero(combined_bins).reshape(-1)
        #
        # pred_fixed = F.one_hot(pred_fixed.cuda().view(1, H, W, D).long())[:, :, :, :, pos]
        # pred_moving = F.one_hot(pred_moving.cuda().view(1, H, W, D).long())[:, :, :, :, pos]
        #
        # weight = 1 / ((torch.bincount(pred_fixed.permute(0, 4, 1, 2, 3).argmax(1).long().reshape(-1)) + torch.bincount(
        #     pred_moving.permute(0, 4, 1, 2, 3).argmax(1).long().reshape(-1))) + eps).float().pow(.3)
        # weight /= weight.mean()
        #
        # features_fix = 10 * (pred_fixed.data.float().permute(0, 4, 1, 2, 3).contiguous() * weight.view(1, -1, 1, 1,
        #                                                                                                1).cuda()).half()
        # features_mov = 10 * (pred_moving.data.float().permute(0, 4, 1, 2, 3).contiguous() * weight.view(1, -1, 1, 1,
        #                                                                                                 1).cuda()).half()
        #
        # return features_fix, features_mov

        print("Not implemented...")

    def _validate_image(self, image: Union[torch.Tensor, np.ndarray, sitk.Image], dtype=float) -> torch.Tensor:
        """Validate image input"""
        if not isinstance(image, torch.Tensor):
            if isinstance(image, sitk.Image):
                image = sitk.GetArrayFromImage(image)
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.astype(dtype))
            else:
                raise ValueError("Input image must be a torch.Tensor, a numpy.ndarray or a SimpleITK.Image")
        return image

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation
        return self.result_transformation_path