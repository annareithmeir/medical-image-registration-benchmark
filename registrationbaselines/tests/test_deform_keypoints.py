import registrationbaselines.core.utils as utils
import unittest
from pathlib import Path
import sys
import socket

import torch

import registrationbaselines.warping.utils_displacement

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.warping.deform_objects
from registrationbaselines.evaluation.plot_objects import plot_all_registration_results
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset

machine_name = socket.gethostname()
if machine_name == "fryderyk":
    path_data = Path("/home/fryderyk/Documents/data/")
elif machine_name == "janus":
    path_data = Path("/data/")
else:
    path_data = Path("/home/anna/datasets/")


class TestKeypoints(unittest.TestCase):

    def test_deform_random_keypoints_zero(self):
        kps_x = torch.rand((10, 3))

        zero_def = torch.zeros((10, 10, 10, 3))
        zero_def = registrationbaselines.warping.utils_displacement.displacement_to_unit_displacement(
            zero_def)

        kps_y = registrationbaselines.warping.deform_objects.deform_keypoints(
            kps_x, zero_def)

        assert (kps_x - kps_y).sum() == 0, (kps_x - kps_y).sum()

    def test_deform_random_keypoints_translation(self):
        kps_x = 10*torch.rand((10, 3))
        kps_x = torch.clamp(kps_x, min=10, max=90)

        zero_def = torch.zeros((100, 100, 100, 3))
        translations = [2, -2, 5]
        zero_def[..., 0] = translations[0]
        zero_def[..., 1] = translations[1]
        zero_def[..., 2] = translations[2]
        zero_def_unit = registrationbaselines.warping.utils_displacement.displacement_to_unit_displacement(
            zero_def)

        kps_y = registrationbaselines.warping.deform_objects.deform_keypoints(
            kps_x, zero_def)
        kps_y_unit = registrationbaselines.warping.deform_objects.deform_keypoints(
            kps_x, zero_def_unit)
        print(kps_x)
        print(kps_y)
        print(kps_y - kps_x)

        # test that deform with non-unit displacement works in the pull direction
        assert torch.all((kps_y[:, 0] - kps_x[:, 0]) == -translations[0])
        assert torch.all((kps_y[:, 1] - kps_x[:, 1]) == -translations[1])
        assert torch.all((kps_y[:, 2] - kps_x[:, 2]) == -translations[2])

        # test that deform with unit displacement works in the pull direction
        assert torch.all((kps_y_unit[:, 0] - kps_x[:, 0]) == -translations[0])
        assert torch.all((kps_y_unit[:, 1] - kps_x[:, 1]) == -translations[1])
        assert torch.all((kps_y_unit[:, 2] - kps_x[:, 2]) == -translations[2])

    def test_deform_loaded_keypoints_zero_and_translation(self):
        kps_file = path_data / "LungCT_preprocessed/keypointsTr/LungCT_0001_0000.csv"
        image_file = path_data / "LungCT_preprocessed/imagesTr/LungCT_0001_0000.nii.gz"
        shape = utils.load_image(image_file).shape
        kps_x = utils.load_keypoints(kps_file)
        assert kps_x.shape[1] == 3

        # deform with zero displacement
        zero_def = torch.zeros((*shape, 3))
        kps_y = registrationbaselines.warping.deform_objects.deform_keypoints(
            kps_x, zero_def)
        assert (kps_x - kps_y).sum() == 0, (kps_x - kps_y).sum()

        # deform with some translation in all directions
        translations = [20, 0, 0]
        zero_def[..., 0] = translations[0]
        zero_def[..., 1] = translations[1]
        zero_def[..., 2] = translations[2]
        zero_def_unit = registrationbaselines.warping.utils_displacement.displacement_to_unit_displacement(
            zero_def)
        kps_y = registrationbaselines.warping.deform_objects.deform_keypoints(
            kps_x, zero_def)
        kps_y_unit = registrationbaselines.warping.deform_objects.deform_keypoints(
            kps_x, zero_def_unit)
        print(kps_x)
        print(kps_y)
        print(kps_y - kps_x)

        # test that deform with non-unit displacement works in the pull direction
        assert torch.all((kps_y[:, 0] - kps_x[:, 0]) == -translations[0])
        assert torch.all((kps_y[:, 1] - kps_x[:, 1]) == -translations[1])
        assert torch.all((kps_y[:, 2] - kps_x[:, 2]) == -translations[2])

        # test that deform with unit displacement works in the pull direction
        assert torch.all((kps_y_unit[:, 0] - kps_x[:, 0]) == -translations[0])
        assert torch.all((kps_y_unit[:, 1] - kps_x[:, 1]) == -translations[1])
        assert torch.all((kps_y_unit[:, 2] - kps_x[:, 2]) == -translations[2])

    def test_deform_keypoints_real_example(self):
        loader = L2RLungCTDataset(path_data / "LungCT_preprocessed",
                                  return_type="torch_tensor_dict",
                                  indices=[0])
        item = loader[0]
        moving_image = item["moving_image"]
        moving_segmentation = item["moving_segmentations"]
        moving_keypoints = item["moving_keypoints"]
        fixed_image = item["fixed_image"]
        fixed_segmentation = item["fixed_segmentations"]
        fixed_keypoints = item["fixed_keypoints"]

        zero_def = torch.zeros((*moving_image.shape, 3))
        zero_def2 = torch.zeros((*moving_image.shape, 3))

        zero_def[..., 2] = 30
        zero_def2[..., 0] = 30

        Warning("It is expected that the deformatoin grid doesn't cover the entire image,\
            as this translation also translates the entire grid")

        zero_def = registrationbaselines.warping.utils_displacement.displacement_to_unit_displacement(
            zero_def)
        deformed_image = registrationbaselines.warping.deform_objects.deform_image(
            moving_image, zero_def)
        deformed_keypoints = registrationbaselines.warping.deform_objects.deform_keypoints(
            moving_keypoints, zero_def2)

        plot_all_registration_results(save_path="registrationbaselines/tests/test_files/temp_vis.png",
                                      moving_image=moving_image,
                                      fixed_image=fixed_image,
                                      pred_image=deformed_image,
                                      pred_segmentations=moving_segmentation,
                                      fixed_segmentations=fixed_segmentation,
                                      moving_keypoints=moving_keypoints,
                                      pred_keypoints=deformed_keypoints,
                                      fixed_keypoints=fixed_keypoints,
                                      displacement=zero_def)
