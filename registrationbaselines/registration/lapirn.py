from registrationbaselines.dl_repos.LapIRN.Code.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_add_lvl2, Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit
from registrationbaselines.dl_repos.LapIRN.Code.Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, transform_unit_flow_to_flow
import os
import torchio as tio
import numpy as np

from pathlib import Path
import torch
from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core.utils_niftyreg import set_intent_code
from registrationbaselines.dl_repos.LapIRN.Code.Functions import save_img, save_flow
from registrationbaselines.core.utils_nifti import transform_nifti_image_with_matrix

import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))

# if we dont do this then LapIRN.Code.miccai2020_model_stage.py can't import Functions
sys.path.append(
    str(Path(__file__).parent.parent.absolute() / "dl_repos/LapIRN/Code"))

class LapIRN(RegistrationInterface):
    """
    Laplacian Image Registration Network (LapIRN, https://github.com/cwmok/LapIRN).
    """
    def __init__(self, path_configuration: Path,
                 dataloader: data_loaders.GenericDataset) -> None:
        """
        Initialize the registration model - inference is performed here.
        """

        self.method = "LapIRN"

        self.config = self.read_config(configuration_path)

        self.fixed_affine = None

        # empty paths
        self.path_fixed = Path()
        self.path_moving = Path()

        # from config
        self.base_dir = Path(__file__).parent.parent.absolute().parent
        self.path_model = self.base_dir / \
            Path(self.config["inference_model_path"])
        self.result_transformed_image_path = self.base_dir / \
            Path(self.config["result_path"]) / 'warped_lapirn.nii.gz'
        self.result_transformation_path = self.base_dir / \
            Path(self.config["result_path"]) / 'disp_lapirn.nii.gz'

        self.device = self.__handle_device_selection()

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        # load moving and fixed images
        moving_img, fixed_img = self._load_images()  # returns torch tensor

        fixed_img = fixed_img.view(
            1, 1, *fixed_img.shape).float().to(self.device)
        moving_img = moving_img.view(
            1, 1, *moving_img.shape).float().to(self.device)
        print(fixed_img.shape)

        imgshape = fixed_img.shape[2:]
        imgshape_2 = tuple(int(x / 2) for x in imgshape)
        imgshape_4 = tuple(int(x / 4) for x in imgshape)

        # load and set up model
        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, self.config["start_channel"], is_train=True, imgshape=imgshape_4,
                                                            range_flow=self.config["range_flow"]).cuda()
        model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, self.config["start_channel"], is_train=True, imgshape=imgshape_2,
                                                            range_flow=self.config["range_flow"], model_lvl1=model_lvl1).cuda()

        model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, self.config["start_channel"], is_train=False, imgshape=imgshape,
                                                       range_flow=self.config["range_flow"], model_lvl2=model_lvl2).cuda()

        transform = SpatialTransform_unit().cuda()

        model.load_state_dict(torch.load(self.path_model))
        model.eval()
        transform.eval()

        grid = generate_grid_unit(imgshape)
        grid = torch.from_numpy(np.reshape(
            grid, (1,) + grid.shape)).cuda().float()

        # predict
        with (torch.no_grad()):
            F_X_Y = model(moving_img, fixed_img)

            X_Y = transform(moving_img, F_X_Y.permute(
                0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.data.cpu().numpy(
            )[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        self.__save_results(X_Y, F_X_Y_cpu)

    def get_transformation_path(self):

        assert self.path_result_transformation.exists(
        ), "Transformation file does not exist."

        return self.path_result_transformation

    def get_transformed_image_path(self):

        assert self.path_result_transformed_image.exists(
        ), "Transformed image file does not exist."

        return self.path_result_transformed_image

    def __handle_device_selection(self) -> str:
        """
        Handle device selection.
        If GPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to the selected GPU and return 'cuda'.
        If CPU is selected, set the CUDA_VISIBLE_DEVICES environment variable to -1 and return 'cpu'.
        """

        num = self.config['gpu']

        if num and (num != '-1'):
            device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = num
        else:
            device = 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        return device

    def _load_images(self):
        subject_dict = {
            "image_m": tio.ScalarImage(self.path_moving),
            "image_f": tio.ScalarImage(self.path_fixed),
        }
        subject = tio.Subject(subject_dict)

        # todo preprocessing
        rescale = tio.RescaleIntensity(
            out_min_max=(0, 1), percentiles=(0, 100))
        subject = rescale(subject)

        self.img_moving = subject["image_m"].data
        self.img_fixed = subject["image_f"].data

        return subject["image_m"].data.squeeze(), subject["image_f"].data.squeeze()

    def __save_results(self, result_transformed_image, result_transformation):

        print("saving disp to ", self.result_transformation_path)
        if not os.path.exists(self.config["result_path"]):
            os.makedirs(self.config["result_path"])

        save_flow(result_transformation, self.result_transformation_path)
        save_img(result_transformed_image, self.result_transformed_image_path)

        affine = np.array([[-1, 0, 0, 0], [0, -1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]])

        transform_nifti_image_with_matrix(
            self.result_transformed_image_path, affine)
        set_intent_code(self.result_transformation_path,
                        'NIFTI_INTENT_DISPVECT')
