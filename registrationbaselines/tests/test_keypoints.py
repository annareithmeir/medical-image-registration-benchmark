import registrationbaselines.core.utils as utils
import unittest
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))


class TestKeypoints(unittest.TestCase):

    def test_deform_random_keypoints_zero(self):
        kps_x = torch.rand((10,3))

        zero_def = torch.zeros((10,10,10,3))
        zero_def = utils.displacement_to_unit_displacement(zero_def)

        kps_y = utils.deform_keypoints(kps_x, zero_def)

        assert (kps_x -kps_y).sum()==0, (kps_x -kps_y).sum()


    def test_deform_random_keypoints_translation(self):
        kps_x = torch.rand((10, 3))

        zero_def = torch.zeros((10, 10, 10, 3))
        zero_def[...,0]=-1
        zero_def = utils.displacement_to_unit_displacement(zero_def)

        kps_y = utils.deform_keypoints(kps_x, zero_def)
        print(kps_x)
        print(kps_y)
        assert (kps_x[:,1:]-kps_y[:,1:]).sum()==0, (kps_x[:,1:]-kps_y[:,1:]) # only first dim changes
        # observation: in y first dim is all smaller values than first dim of x for positive displacement
        # and larger for negative displacement
