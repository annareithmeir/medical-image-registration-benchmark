import unittest
from pathlib import Path
import sys
import socket

import torch

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset
from registrationbaselines.core.visualization import plot_all_registration_results
import registrationbaselines.core.utils as utils

machine_name = socket.gethostname()
if machine_name == "fryderyk":
    path_data = Path("/home/fryderyk/Documents/data/")
elif machine_name == "janus":
    path_data = Path("/data/")
else:
    path_data = Path("/home/anna/datasets/")


class TestVisualization(unittest.TestCase):

    def test_visualization_dataloader(self):
        loader = L2RLungCTDataset(path_data / "LungCT_preprocessed",
                                  return_type="torch_tensor_dict",
                                  indices=[0])
        loader.plot_random_image()
        pass

    def test_visualization(self):
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
        zero_def = utils.displacement_to_unit_displacement(zero_def)
        plot_all_registration_results(save_path="registrationbaselines/tests/test_files/temp_vis.png",
                                      moving_image=moving_image,
                                      fixed_image=fixed_image,
                                      pred_image=moving_image,
                                      pred_segmentations=moving_segmentation,
                                      fixed_segmentations=fixed_segmentation,
                                      moving_keypoints=moving_keypoints,
                                      pred_keypoints=moving_keypoints,
                                      fixed_keypoints=fixed_keypoints,
                                      displacement=zero_def)
