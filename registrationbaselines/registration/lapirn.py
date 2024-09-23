import numpy as np
from pathlib import Path
import sys

from typing import Tuple

import torch

from registrationbaselines.core import utils_dl
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.interfaces._interface_registration import RegistrationInterface

# sys.path.append(str(Path(__file__).parent.absolute().parent))
# if we dont do this then LapIRN.Code.miccai2020_model_stage.py can't import Functions

sys.path.append(
    str(Path(__file__).parent.parent.absolute() / "dl_repos/LapIRN/Code"))  # nopep8
from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, \
Miccai2020_LDR_laplacian_unit_add_lvl2, Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit  # nopep8
from registrationbaselines.dl_repos.LapIRN.Code.Functions import generate_grid_unit, transform_unit_flow_to_flow  # nopep8


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
                         use_masked_evaluation,
                         model_path)

    def _register(self,
                  fixed_image: torch.Tensor,
                  moving_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        original_shape = list(fixed_image.shape)

        fixed_image = fixed_image.view(
            1, 1, *fixed_image.shape).float().to(self.device)
        moving_image = moving_image.view(
            1, 1, *moving_image.shape).float().to(self.device)

        new_shape = utils_dl.get_new_lapirn_image_shape(original_shape)

        image_shape_1 = new_shape
        image_shape_2 = tuple(int(x / 2) for x in image_shape_1)
        image_shape_4 = tuple(int(x / 4) for x in image_shape_1)

        moving_image = utils_dl.pad_tensor_to_shape(
            moving_image, new_shape)
        fixed_image = utils_dl.pad_tensor_to_shape(fixed_image, new_shape)

        # load and set up model
        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.run_configuration["start_channel"], is_train=True, imgshape=image_shape_4,
                                                            range_flow=self.run_configuration["range_flow"]).cuda()
        model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, self.run_configuration["start_channel"], is_train=True, imgshape=image_shape_2,
                                                            range_flow=self.run_configuration["range_flow"], model_lvl1=model_lvl1).cuda()

        model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, self.run_configuration["start_channel"], is_train=False, imgshape=image_shape_1,
                                                       range_flow=self.run_configuration["range_flow"], model_lvl2=model_lvl2).cuda()

        transform = SpatialTransform_unit().cuda()

        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        transform.eval()

        grid = generate_grid_unit(image_shape_1)
        grid = torch.from_numpy(np.reshape(
            grid, (1,) + grid.shape)).cuda().float()

        # predict
        with torch.no_grad():
            F_X_Y = model(moving_image, fixed_image)
            X_Y = transform(moving_image, F_X_Y.permute(
                0, 2, 3, 4, 1), grid).data.cpu()[0, 0, :, :, :]
            F_X_Y_cpu = F_X_Y.data.cpu()[0, :, :, :, :].permute(1, 2, 3, 0)

            deformed_image = utils_dl.crop_tensor_to_shape(X_Y,
                                                           list(original_shape))  # .permute(2,1,0)
            displacement = utils_dl.crop_tensor_to_shape(F_X_Y_cpu,
                                                         list(original_shape) + [3])

            return deformed_image, displacement
