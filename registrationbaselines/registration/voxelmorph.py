import registrationbaselines.dl_repos.voxelmorph.voxelmorph as vxm
from pathlib import Path
import sys
import os

import torch

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))


class VoxelmorphReg(RegistrationInterface):
    def __init__(self, configuration_path: Path):
        """
        Initialize the registration model - inference is performed here.
        """

        self.method = "VoxelMorph"

        self.configuration = self.read_config(configuration_path)

        self.fixed_affine = None

        # empty paths
        self.path_fixed = Path()
        self.path_moving = Path()
        self.path_result_transformed_image = Path()
        self.path_result_transformation = Path()

        # from config
        self.path_model = Path(self.configuration["model_path"])
        self.device = self.__handle_device_selection()

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        # load moving and fixed images
        add_feat_axis = not self.configuration['multichannel']
        moving = vxm.py.utils.load_volfile(
            self.path_moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed, self.fixed_affine = vxm.py.utils.load_volfile(
            self.path_fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

        # load and set up model
        model = vxm.torch.networks.VxmDense.load(
            self.configuration['model_path'], self.device)
        model.to(self.device)
        model.eval()

        # set up tensors and permute
        input_moving = torch.from_numpy(moving).to(
            self.device).float().permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(fixed).to(
            self.device).float().permute(0, 4, 1, 2, 3)

        # predict
        moved, warp = model(input_moving, input_fixed, registration=True)

        moved = moved.detach().cpu().numpy().squeeze()
        warp = warp.detach().cpu().numpy().squeeze()

        self.__save_results(moved, warp)

    def get_transformation_path(self):

        assert self.path_result_transformation.exists(
        ), "Transformation file does not exist."

        return self.path_result_transformation

    def get_transformed_image_path(self):

        assert self.path_result_transformed_image.exists(
        ), "Transformed image file does not exist."

        return self.path_result_transformed_image

    def __save_results(self, deformed, deformation):
        self.path_result_transformed_image, self.path_result_transformation = \
            utils_commandline.create_result_paths(self.path_fixed.parent,
                                                  self.path_fixed.stem,
                                                  self.path_moving.stem,
                                                  self.method,
                                                  ".nii",
                                                  ".nii")

        vxm.py.utils.save_volfile(
            deformed, self.path_result_transformed_image, self.fixed_affine)
        vxm.py.utils.save_volfile(
            deformation, self.path_result_transformation, self.fixed_affine)

    def __handle_device_selection(self) -> str:
        """
        Handle device selection.
        If GPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to the selected GPU and return 'cuda'.
        If CPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to -1 and return 'cpu'.
        """

        num = self.configuration['gpu_number']

        if num and (num != '-1'):
            device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = num
        else:
            device = 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        return device
