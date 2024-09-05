from pathlib import Path
import sys
import os

import registrationbaselines.dl_repos.voxelmorph.voxelmorph as vxm
from registrationbaselines.core import utils_dl
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.interfaces._interface_registration import RegistrationInterface
from registrationbaselines.io import load
from registrationbaselines.warping import utils_displacement

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))


class VoxelMorph(RegistrationInterface):
    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 model_path: Path,
                 use_masked_evaluation: bool = True) -> None:
        """
        Initialize the registration model - inference is performed here.
        """

        super().__init__("VoxelMorph",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation,
                         model_path)

        self.gpu_number = self.general_configuration["parameters"]['gpu_number']["values"][0]
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
        padded_shape = utils_dl.get_new_voxelmorph_image_shape(list(fixed.shape),
                                                               self.number_of_layers)
        fixed.unsqueeze_(0).unsqueeze_(0)
        moving.unsqueeze_(0).unsqueeze_(0)
        fixed = utils_dl.pad_tensor_to_shape(fixed, padded_shape)
        moving = utils_dl.pad_tensor_to_shape(moving, padded_shape)

        # load and set up model
        model = vxm.torch.networks.VxmDense.load(self.model_path, self.device)
        model.to(self.device)
        model.eval()

        # predict
        warped, displacement = model(moving,
                                     fixed,
                                     registration=True)

        # convert tensors to a format accepted by our framework, by cropping
        warped = warped.detach().cpu().squeeze()
        warped = utils_dl.crop_tensor_to_shape(warped, list(ori_shape))

        displacement = displacement.detach().cpu().squeeze()

        displacement = displacement.permute(1, 2, 3, 0)
        displacement = displacement[..., [2, 1, 0]]
        displacement = utils_displacement.displacement_to_unit_displacement(
            displacement)

        displacement = utils_dl.crop_tensor_to_shape(
            displacement, list(ori_shape) + [3])

        self._save_results(warped, displacement)
