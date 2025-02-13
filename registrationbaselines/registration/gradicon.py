from pathlib import Path
import sys
import os

from typing import Tuple

import itk
import torch

from registrationbaselines.core import utils_dl
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.io import load
from registrationbaselines.displacement import utils_displacement


sys.path.append(str(Path(__file__).parent.absolute().parent))


class GradICON(RegistrationInterface):
    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 model_path: Path,
                 use_masked_evaluation: bool = True) -> None:
        """
        Initialize the registration model - inference is performed here.
        """

        super().__init__("GradICON",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation,
                         model_path)

        self.gpu_number = self.general_configuration["parameters"]['gpu_number']["values"][0]

    def _register(self,
                  fixed_image: torch.Tensor,
                  moving_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        ori_shape = fixed_image.shape

        # convert tensors to shapes accepted by voxelmorph, by padding
        new_shape = [4 * 40, 4 * 96, 4 * 96]

        fixed_image = utils_dl.reshape_tensor(fixed_image, new_shape)
        moving_image = utils_dl.reshape_tensor(moving_image, new_shape)

        fixed_image.unsqueeze_(0).unsqueeze_(0)
        moving_image.unsqueeze_(0).unsqueeze_(0)

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        model = utils_dl.make_gradicon_network([1, 1] + list(new_shape),
                                               1)

        checkpoint = torch.load(self.model_path,
                                map_location=self.device)
        checkpoint = {
            key.replace("regis_net.", ""): value
            for key, value in checkpoint.items()
        }
        model.regis_net.load_state_dict(checkpoint)
        model.eval()

        fixed_itk = itk.image_from_array(
            fixed_image.cpu().detach().numpy().squeeze())
        moving_itk = itk.image_from_array(
            moving_image.cpu().detach().numpy().squeeze())

        # predict
        phi_AB, phi_BA = model(fixed_itk, moving_itk)

        x = 0

        # convert tensors to a format accepted by our framework, by cropping
        # warped = warped.detach().cpu().squeeze()
        # deformed_image = utils_dl.crop_tensor_to_shape(warped, list(ori_shape))

        # displacement = displacement.detach().cpu().squeeze()

        # displacement = displacement.permute(1, 2, 3, 0)
        # displacement = displacement[..., [2, 1, 0]]

        # displacement = utils_displacement.displacement_to_unit_displacement(
        #     displacement)

        # displacement = utils_displacement.displacement_to_unit_displacement(
        #     displacement)

        # displacement = utils_dl.crop_tensor_to_shape(
        #     displacement, list(ori_shape) + [3])

        # return deformed_image, displacement
