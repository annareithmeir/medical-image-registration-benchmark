import unittest
from pathlib import Path
import sys

import numpy as np
import torch
import SimpleITK as sitk

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.core.utils as utils

"""
Should utils.save_image return spacing?
"""


class TestSaveLoad(unittest.TestCase):

    def test_repeatedly_save_and_load(self):

        image_to_save = torch.rand(10, 10, 10)
        spacing = (1, 1, 1)
        file_path = Path("registrationbaselines/tests/test_files/tmp.nii.gz")

        for _ in range(10):

            utils.save_image(image_to_save, file_path, spacing)
            image_loaded = utils.load_image(file_path)

            self.assertAlmostEqual(image_to_save.sum().detach().cpu().numpy(),
                                   image_loaded.sum().detach().cpu().numpy(),
                                   delta=1e-4)
            self.assertAlmostEqual(image_to_save.detach().cpu().numpy().shape,
                                   image_loaded.detach().cpu().numpy().shape,
                                   delta=0.0)

            image_to_save = image_loaded

        file_path.unlink()
