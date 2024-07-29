import unittest
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset
import registrationbaselines.core.utils as utils
from registrationbaselines.core.visualization import plot_all_registration_results

class TestDeformImage(unittest.TestCase):

    def test_deform_random_image_zero(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        x = torch.rand(2,2,2)
        zero_def = torch.zeros(*x.shape, 3)
        zero_def=utils.displacement_to_unit_displacement(zero_def)
        y=utils.deform_image(x, zero_def)
        print(x)
        print(y)
        print(zero_def)
        assert (x-y).sum() == 0, (x-y).sum()
        assert x.shape == y.shape

    def test_deform_random_image_displacement(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        x = torch.rand(2,2,2)
        zero_def = torch.zeros(*x.shape, 3)
        zero_def[...,0]=1
        zero_def=utils.displacement_to_unit_displacement(zero_def)
        y=utils.deform_image(x, zero_def)
        print(x)
        print(y)
        print(zero_def)
        assert (x[..., 1:] - y[..., :2]).sum() == 0
        assert x.shape == y.shape

    def test_deform_loaded_image_zero(self):
        """
        This test tests the zero displacement of a image with utils.deform_image()
        @return:
        """

        loader = L2RLungCTDataset(Path("/home/anna/datasets/LungCT_preprocessed"), return_type="path_dict",
                                  indices=[0])
        item = loader[0]
        x = utils.load_image(item["moving_image"])
        zero_def = torch.zeros(*x.shape, 3)
        zero_def=utils.displacement_to_unit_displacement(zero_def)
        y=utils.deform_image(x, zero_def)
        # print(x)
        # print(y)
        # print(zero_def)

        plot_all_registration_results(save_path=None,
                                      moving_image=x,
                                      fixed_image=y,
                                      pred_image=y,
                                      displacement=zero_def)

        assert (x-y).sum() == 0, (x-y).sum()
        assert x.shape == y.shape

    def test_deform_random_segmentation_zero(self):
        """
        This test tests the zero displacement of a segmentation map with utils.deform_image()
        @return:
        """
        x = torch.zeros((3, 3, 3), dtype=torch.int16)
        x[0,:]=1
        x[2,:]=2
        zero_def = torch.zeros(*x.shape, 3)
        zero_def=utils.displacement_to_unit_displacement(zero_def)
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

        x = torch.zeros((3, 3, 3), dtype=torch.int16)
        x[0, :] = 1
        x[2, :] = 2
        zero_def = torch.zeros(*x.shape, 3)
        zero_def[...,0]=1
        zero_def=utils.displacement_to_unit_displacement(zero_def)
        y=utils.deform_image(x, zero_def)
        print(x)
        print(y)
        print(x[..., 1:])
        print(y[..., :2])
        print(zero_def)
        assert (x[..., 1:] - y[..., :2]).sum() == 0 # in last dim, all entries have moved by 1
        assert x.shape == y.shape

    def test_deform_loaded_segmentation_zero(self):
        """
        This test tests the zero displacement of a real segmentation map with utils.deform_image()
        @return:
        """

        loader = L2RLungCTDataset(Path("/home/anna/datasets/LungCT_preprocessed"), return_type="path_dict",
                                  indices=[0])
        item = loader[0]
        x = utils.load_image(item["moving_segmentations"])
        print(x.dtype)
        x = x.short()
        zero_def = torch.zeros(*x.shape, 3)
        zero_def=utils.displacement_to_unit_displacement(zero_def)
        y=utils.deform_image(x, zero_def)

        plot_all_registration_results(save_path=None,
                                      moving_image=x,
                                      fixed_image=y,
                                      pred_image=y,
                                      displacement=zero_def)

        assert (x-y).sum() == 0, (x-y).sum()
        assert x.shape == y.shape

    def test_deform_image_niftyreg(self):
        pass
