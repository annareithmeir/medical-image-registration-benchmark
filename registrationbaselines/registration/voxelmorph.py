from pathlib import Path
import sys
import os

import registrationbaselines.dl_repos.voxelmorph.voxelmorph as vxm
from registrationbaselines.core import utils_voxelmorph
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.io import load

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))


class VoxelMorph(RegistrationInterface):
    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True):
        """
        Initialize the registration model - inference is performed here.
        """

        super().__init__("VoxelMorph",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation)

        # from config
        self.path_model = Path(
            self.general_configuration["parameters"]["model_path"]["values"][0])

        self.gpu_number = self.general_configuration["parameters"]['gpu_number']["values"][0]
        self.device = self.__handle_device_selection()

        self.number_of_layers = len(
            self.general_configuration["parameters"]['enc']["values"][0])

    def _register(self,
                  fixed_image_path: Path,
                  moving_image_path: Path) -> None:

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        # load moving and fixed images
        fixed = load.load_image(self.path_fixed).to(self.device)
        moving = load.load_image(self.path_moving).to(self.device)
        ori_shape = moving.shape

        # convert tensors to shapes accepted by voxelmorph, by padding
        padded_shape = utils_voxelmorph.get_new_voxelmorph_image_shape(list(fixed.shape),
                                                                       self.number_of_layers)
        fixed.unsqueeze_(0).unsqueeze_(0)
        moving.unsqueeze_(0).unsqueeze_(0)
        fixed = utils_voxelmorph.pad_tensor_to_shape(fixed, padded_shape)
        moving = utils_voxelmorph.pad_tensor_to_shape(moving, padded_shape)

        # load and set up model
        model = vxm.torch.networks.VxmDense.load(self.path_model, self.device)
        model.to(self.device)
        model.eval()

        # predict
        warped, displacement = model(moving,
                                     fixed,
                                     registration=True)

        # convert tensors to a format accepted by our framework, by cropping
        warped = warped.detach().cpu().squeeze()
        warped = utils_voxelmorph.crop_tensor_to_shape(warped, list(ori_shape))
        displacement = displacement.detach().cpu().squeeze()
        displacement = displacement.permute(1, 2, 3, 0)
        displacement = utils_voxelmorph.crop_tensor_to_shape(
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
