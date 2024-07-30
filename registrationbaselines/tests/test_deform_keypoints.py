import registrationbaselines.core.utils as utils
import unittest
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))
from registrationbaselines.core.visualization import plot_all_registration_results
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset

class TestKeypoints(unittest.TestCase):

    def test_deform_random_keypoints_zero(self):
        kps_x = torch.rand((10,3))

        zero_def = torch.zeros((10,10,10,3))
        zero_def = utils.displacement_to_unit_displacement(zero_def)

        kps_y = utils.deform_keypoints(kps_x, zero_def)

        assert (kps_x -kps_y).sum()==0, (kps_x -kps_y).sum()


    def test_deform_random_keypoints_translation(self):
        kps_x = 10*torch.rand((10, 3))
        kps_x = torch.clamp(kps_x, min=10, max=90)

        zero_def = torch.zeros((100, 100, 100, 3))
        zero_def[...,0]=2
        zero_def[...,1]=-2
        zero_def[...,2]=5
        zero_def_unit = utils.displacement_to_unit_displacement(zero_def)

        kps_y = utils.deform_keypoints(kps_x, zero_def)
        kps_y_unit = utils.deform_keypoints(kps_x, zero_def_unit)
        print(kps_x)
        print(kps_y)
        print(kps_y -kps_x)

        # test that deform with non-unit displacement works in the pull direction
        assert torch.all((kps_y[:, 0] - kps_x[:, 0]) == 2)
        assert torch.all((kps_y[:, 1] - kps_x[:, 1]) == -2)
        assert torch.all((kps_y[:, 2] - kps_x[:, 2]) == 5)

        # test that deform with unit displacement works in the pull direction
        assert torch.all((kps_y_unit[:, 0] - kps_x[:, 0]) == 2)
        assert torch.all((kps_y_unit[:, 1] - kps_x[:, 1]) == -2)
        assert torch.all((kps_y_unit[:, 2] - kps_x[:, 2]) == 5)


    def test_deform_loaded_keypoints_zero_and_translation(self):
        kps_file = Path("/home/anna/datasets/LungCT/keypointsTr/LungCT_0001_0000.csv")
        kps_x = utils.load_keypoints(kps_file)
        assert kps_x.shape[1]==3

        # deform with zero displacement
        zero_def = torch.zeros((100, 100, 100, 3))
        kps_y = utils.deform_keypoints(kps_x, zero_def)
        assert (kps_x - kps_y).sum() == 0, (kps_x - kps_y).sum()

        # deform with some translation in all directions
        translations=[20,0,0]
        zero_def[...,0]=translations[0]
        zero_def[...,1]=translations[1]
        zero_def[...,2]=translations[2]
        zero_def_unit = utils.displacement_to_unit_displacement(zero_def)
        kps_y = utils.deform_keypoints(kps_x, zero_def)
        kps_y_unit = utils.deform_keypoints(kps_x, zero_def_unit)
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
        loader = L2RLungCTDataset(Path("/home/anna/datasets/LungCT_preprocessed"), return_type="torch_tensor_dict",
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

        zero_def[...,2]=30
        zero_def2[...,0]=30
        #zero_def = utils.displacement_to_unit_displacement(zero_def)
        deformed_image = utils.deform_image(moving_image, zero_def)
        deformed_keypoints = utils.deform_keypoints(moving_keypoints, zero_def2)

        plot_all_registration_results(save_path=None,
                                      moving_image=moving_image,
                                      fixed_image=fixed_image,
                                      pred_image=deformed_image,
                                      pred_segmentations=moving_segmentation,
                                      fixed_segmentations=fixed_segmentation,
                                      moving_keypoints=moving_keypoints,
                                      pred_keypoints=deformed_keypoints,
                                      fixed_keypoints=fixed_keypoints,
                                      displacement=zero_def)


