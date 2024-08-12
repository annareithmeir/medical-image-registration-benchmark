import registrationbaselines.dl_repos.voxelmorph.voxelmorph as vxm
from pathlib import Path
import sys
import os

import torch

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils
from registrationbaselines.data_loading import data_loaders

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))


class VoxelMorphReg(RegistrationInterface):
    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset):
        """
        Initialize the registration model - inference is performed here.
        """

        self.method_name = "VoxelMorph"

        self.configuration = utils.read_config(configuration_path)

        self.dataloader = dataloader

        # from config
        self.path_model = Path(
            self.configuration["parameters"]["model_path"]["values"][0])

        self.gpu_number = self.configuration["parameters"]['gpu_number']["values"][0]
        self.device = self.__handle_device_selection()

        self.number_of_layers = len(
            self.configuration["parameters"]['enc']["values"][0])

    def _register(self,
                  fixed_image_path: Path,
                  moving_image_path: Path) -> None:

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        # load moving and fixed images
        fixed = utils.load_image(self.path_fixed).to(self.device)
        moving = utils.load_image(self.path_moving).to(self.device)
        ori_shape = moving.shape
        ori_fixed = fixed.detach().clone()

        padded_shape = utils.get_new_voxelmorph_image_shape(list(fixed.shape),
                                                            self.number_of_layers)
        fixed.unsqueeze_(0).unsqueeze_(0)
        moving.unsqueeze_(0).unsqueeze_(0)
        fixed = utils.pad_tensor_to_shape(fixed, padded_shape)
        moving = utils.pad_tensor_to_shape(moving, padded_shape)

        fixed_cropped = utils.crop_tensor_to_shape(
            fixed.squeeze(), list(ori_shape))

        a = torch.allclose(fixed_cropped, ori_fixed)

        # load and set up model
        model = vxm.torch.networks.VxmDense.load(self.path_model, self.device)
        model.to(self.device)
        model.eval()

        # predict
        warped, displacement = model(moving,
                                     fixed,
                                     registration=True)

        warped = warped.detach().cpu().squeeze()
        warped = utils.crop_tensor_to_shape(warped, list(ori_shape))
        displacement = displacement.detach().cpu().squeeze()
        displacement = displacement.permute(1, 2, 3, 0)
        displacement = utils.crop_tensor_to_shape(
            displacement, list(ori_shape) + [3])

        self._save_results(warped, displacement)

    def __handle_device_selection(self) -> str:
        """
        Handle device selection.
        If GPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to the selected GPU and return 'cuda'.
        If CPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to -1 and return 'cpu'.
        """

        if self.gpu_number and (self.gpu_number != '-1'):
            device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_number
        else:
            device = 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        return device
