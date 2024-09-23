import unittest
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.io import load, save


class TestDataloaders(unittest.TestCase):

    def test_save_displacement(self):

        x = np.random.rand(10, 10, 10, 3)
        tmp_filename = Path("/home/anna/tmp.nii.gz")
        save.save_image(x, tmp_filename)
        y = load.load_image(tmp_filename)
        assert (x-y).sum() == 0
        assert x.shape == y.shape

        z = sitk.ReadImage(tmp_filename, sitk.sitkVectorFloat64)
        print(z.GetSize())
