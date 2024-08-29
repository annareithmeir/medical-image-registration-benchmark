import unittest
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk

import registrationbaselines.io.io

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))


class TestDataloaders(unittest.TestCase):

    def test_save_displacement(self):

        x = np.random.rand(10, 10, 10, 3)
        tmp_filename = Path("/home/anna/tmp.nii.gz")
        registrationbaselines.io.io.save_image(x, tmp_filename)
        y = registrationbaselines.io.io.load_image(tmp_filename)
        assert (x-y).sum() == 0
        assert x.shape == y.shape

        z = sitk.ReadImage(tmp_filename, sitk.sitkVectorFloat64)
        print(z.GetSize())
