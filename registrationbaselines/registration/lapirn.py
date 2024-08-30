from pathlib import Path

from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_add_lvl2, Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit
from registrationbaselines.dl_repos.LapIRN.Code.Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, transform_unit_flow_to_flow

import numpy as np
import torch

from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.core import utils_nifti
from registrationbaselines.io import load

import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))

# if we dont do this then LapIRN.Code.miccai2020_model_stage.py can't import Functions
sys.path.append(
    str(Path(__file__).parent.parent.absolute() / "dl_repos/LapIRN/Code"))


class LapIRN(RegistrationInterface):
    """
    Laplacian Image Registration Network (LapIRN, https://github.com/cwmok/LapIRN).
    """

    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 model_path: Path,
                 use_masked_evaluation: bool = True) -> None:
        """
        Initialize the registration model - inference is performed here.
        """

        super().__init__("LapIRN",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation)

        self.fixed_affine = None

        # from config
        self.base_dir = Path(__file__).parent.parent.absolute().parent

        self.path_model = model_path

        self.device = self._handle_device_selection()

    def _register(self, fixed_image_path: Path, moving_image_path: Path) -> None:

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        # load moving and fixed images
        fixed_img = load.load_image(self.path_fixed)
        moving_img = load.load_image(self.path_moving)

        fixed_img = fixed_img.view(
            1, 1, *fixed_img.shape).float().to(self.device)
        moving_img = moving_img.view(
            1, 1, *moving_img.shape).float().to(self.device)

        imgshape = fixed_img.shape[2:]
        imgshape_2 = tuple(int(x / 2) for x in imgshape)
        imgshape_4 = tuple(int(x / 4) for x in imgshape)

        # load and set up model
        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3,
                                                            self.run_configuration["start_channel"],
                                                            is_train=True,
                                                            imgshape=imgshape_4,
                                                            range_flow=self.run_configuration["range_flow"]).cuda()

        model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3,
                                                            self.run_configuration["start_channel"],
                                                            is_train=True,
                                                            imgshape=imgshape_2,
                                                            range_flow=self.run_configuration["range_flow"],
                                                            model_lvl1=model_lvl1).cuda()

        model = Miccai2020_LDR_laplacian_unit_add_lvl3(2,
                                                       3,
                                                       self.run_configuration["start_channel"],
                                                       is_train=False,
                                                       imgshape=imgshape,
                                                       range_flow=self.run_configuration["range_flow"],
                                                       model_lvl2=model_lvl2).cuda()

        transform = SpatialTransform_unit().cuda()

        model.load_state_dict(torch.load(self.path_model))
        model.eval()
        transform.eval()

        grid = generate_grid_unit(imgshape)
        grid = torch.from_numpy(np.reshape(
            grid, (1,) + grid.shape)).cuda().float()

        # predict
        with torch.no_grad():
            F_X_Y = model(moving_img, fixed_img)

            X_Y = transform(moving_img, F_X_Y.permute(
                0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.data.cpu().numpy(
            )[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        self._save_results(X_Y, F_X_Y_cpu)

        # todo - do we need this?
        affine = np.array([[-1, 0, 0, 0], [0, -1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        utils_nifti.transform_nifti_image_with_matrix(
            self.path_result_deformed, affine)
