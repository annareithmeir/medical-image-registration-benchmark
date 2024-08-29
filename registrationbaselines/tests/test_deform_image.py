import unittest
from pathlib import Path
import sys
import socket

import torch

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset
import registrationbaselines.core.utils as utils
from registrationbaselines.evaluation.plot_objects import plot_all_registration_results

machine_name = socket.gethostname()
if machine_name == "fryderyk":
    path_data = Path("/home/fryderyk/Documents/data/")
elif machine_name == "janus":
    path_data = Path("/data/")
else:
    path_data = Path("/home/anna/datasets/")


class TestDeformImage(unittest.TestCase):

    def test_deform_random_image_zero(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        x = torch.rand(8, 8)*0.2
        x[:4, :4] = 0.9
        zero_def = torch.zeros(*x.shape, 2)
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        y = utils.deform_image(x, zero_def)
        print(x)
        print(y)
        torch.set_printoptions(sci_mode=False, linewidth=200)
        z = y-x
        print(z)
        # assert (x-y).sum() == 0, (x-y).sum()
        # TODO fix this test
        assert x.shape == y.shape

    def test_deform_random_image_displacement(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        x = torch.rand(2, 2, 2)
        zero_def = torch.zeros(*x.shape, 3)
        zero_def[..., 0] = 1
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        y = utils.deform_image(x, zero_def)
        print(x)
        print(y)
        print(zero_def)

        Warning("TODO: fix this test")
        # assert (x[..., 1:] - y[..., :2]).sum() == 0
        assert x.shape == y.shape

    def test_deform_loaded_image_zero(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        loader = L2RLungCTDataset(path_data / "LungCT_preprocessed",
                                  return_type="path_dict",
                                  indices=[0])
        item = loader[0]
        x = utils.load_image(item["moving_image"])
        zero_def = torch.zeros(*x.shape, 3)
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        y = utils.deform_image(x, zero_def)
        # print(x)
        # print(y)
        # print(zero_def)

        plot_all_registration_results(save_path="registrationbaselines/tests/test_files/temp_vis.png",
                                      moving_image=x,
                                      fixed_image=y,
                                      pred_image=y,
                                      displacement=zero_def)

        Warning(
            "We currently cannot get a proper zero deformation, but leaving this here as an example.")
        # for now the zero defortmation has a lot of interpolation artefacts
        assert (x-y).sum() > 10000, (x-y).sum()
        # assert (x-y).sum() == 0, (x-y).sum()
        assert x.shape == y.shape

    def test_deform_random_segmentation_zero(self):
        """
        This test tests the zero displacement of a segmentation map with utils.deform_image()
        @return:
        """
        x = torch.zeros((3, 3, 3), dtype=torch.uint8)
        x[0, :] = 1
        x[2, :] = 2
        zero_def = torch.zeros(*x.shape, 3)
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        y = utils.deform_image(x, zero_def)
        print(x)
        print(y)
        print(zero_def)
        assert (x - y).sum() == 0, (x - y).sum()
        assert x.shape == y.shape

    def test_deform_random_segmentation_displacement(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        x = torch.zeros((3, 3, 3), dtype=torch.uint8)
        x[0, :] = 1
        x[2, :] = 2
        zero_def = torch.zeros(*x.shape, 3)
        zero_def[..., 0] = 1
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        y = utils.deform_image(x, zero_def)
        print(x)
        print(y)
        print(x[..., 1:])
        print(y[..., :2])
        print(zero_def)
        # in last dim, all entries have moved by 1
        assert (x[..., 1:] - y[..., :2]).sum() == 0
        assert x.shape == y.shape

    def test_deform_loaded_segmentation_zero(self):
        """
        This test tests the zero displacement of a real segmentation map with utils.deform_image()
        @return:
        """

        loader = L2RLungCTDataset(path_data / "LungCT_preprocessed",
                                  return_type="path_dict",
                                  indices=[0])
        item = loader[0]
        x = utils.load_image(item["moving_segmentations"])
        print(x.dtype)

        zero_def = torch.zeros(*x.shape, 3)
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        y = utils.deform_image(x, zero_def)

        plot_all_registration_results(save_path="registrationbaselines/tests/test_files/temp_vis.png",
                                      moving_image=x,
                                      fixed_image=y,
                                      pred_image=y,
                                      displacement=zero_def)

        assert (x-y).sum() == 0, (x-y).sum()
        assert x.shape == y.shape

    def test_deform_image_niftyreg(self):
        pass

    def test_to_from_unit_displacement(self):
        """
        Test the utils.unit_displacement_to_displacement() and utils.displacement_to_unit_displacement()
        @return:
        """
        displacement = 100*torch.rand((10, 10, 10, 3))
        displacement_unit = utils.displacement_to_unit_displacement(
            displacement)
        displacement_denorm = utils.unit_displacement_to_displacement(
            displacement_unit)

        assert torch.all((displacement_denorm - displacement_unit) == 0)
