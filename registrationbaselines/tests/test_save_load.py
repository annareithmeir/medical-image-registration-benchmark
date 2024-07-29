import unittest
from pathlib import Path
import sys

import numpy as np
import torch
import SimpleITK as sitk

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.core.utils as utils
import registrationbaselines.core.utils_nifti as utils_nifti

"""
TODO
dataloader preprocess has to save segmentations as np.uint16 and not np.uint8 as it is currently done

DISCUSS
Should utils.save_image return spacing?
Remove default arugment from deform_image - and just check for float or int and raise otherwise
Maybe lets do a laod_image, load_semgentation, load_displacement functions? - same for save
Should our laod and save functions preprocess data if they notice they are not correct? like unit displacement
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

    def test_load_image_interface(self):

        # check that load_image returns a tensor
        file_path = Path(
            "registrationbaselines/tests/test_files/LungCT_0001_0000_preprocessed_segmentation.nii.gz")

        image = utils.load_image(file_path)
        self.assertTrue(isinstance(image, torch.Tensor))

        # check that it raises if file is not nifti but file exists
        file_path = Path(
            "registrationbaselines/tests/test_files/results_all.csv")
        self.assertRaises(ValueError, utils.load_image, file_path)

        # check that it raises if file is not nifti and file does not exist
        file_path = Path(
            "registrationbaselines/tests/test_files/does_not_exist.txt")
        self.assertRaises(ValueError, utils.load_image, file_path)

        # check that it raises if file is nifti and file does not exist
        file_path = Path(
            "registrationbaselines/tests/test_files/does_not_exist.nii.gz")
        self.assertRaises(FileNotFoundError, utils.load_image, file_path)

        # load 5D image and check that it raises
        file_path_5D = Path(
            "registrationbaselines/tests/test_files/image_tmp_5D.nii.gz")
        array_5D = np.random.rand(10, 10, 10, 10, 10).astype(np.float32)
        sitk_image_5D = sitk.GetImageFromArray(array_5D)
        sitk.WriteImage(sitk_image_5D, file_path_5D)
        self.assertRaises(ValueError, utils.load_image, file_path_5D)
        file_path_5D.unlink()

        # load an image in wrong dtype
        file_path_wrong_dtype = Path(
            "registrationbaselines/tests/test_files/image_tmp.nii.gz")
        array_wrong_dtype = np.random.rand(10, 10, 10).astype(np.float64)
        sitk_image_wrong_dtype = sitk.GetImageFromArray(array_wrong_dtype)
        sitk.WriteImage(sitk_image_wrong_dtype, file_path_wrong_dtype)

        self.assertRaises(TypeError, utils.load_image, file_path_wrong_dtype)

        file_path_wrong_dtype.unlink()

    def test_load_displacement_interface(self):

        # check that load_image returns a tensor
        file_path = Path(
            "registrationbaselines/tests/test_files/LungCT_0001_0001_deformation_to_LungCT_0001_0000.nii.gz")

        Warning("restore this test")
        image = utils.load_displacement(file_path)
        self.assertTrue(isinstance(image, torch.Tensor))

        # check that it raises if file is not nifti but file exists
        file_path = Path(
            "registrationbaselines/tests/test_files/results_all.csv")
        self.assertRaises(ValueError, utils.load_displacement, file_path)

        # check that it raises if file is not nifti and file does not exist
        file_path = Path(
            "registrationbaselines/tests/test_files/does_not_exist.txt")
        self.assertRaises(ValueError, utils.load_displacement, file_path)

        # check that it raises if file is nifti and file does not exist
        file_path = Path(
            "registrationbaselines/tests/test_files/does_not_exist.nii.gz")
        self.assertRaises(FileNotFoundError,
                          utils.load_displacement, file_path)

        # load 5D image and check that it raises
        file_path_5D = Path(
            "registrationbaselines/tests/test_files/image_tmp_5D.nii.gz")
        array_5D = np.random.rand(10, 10, 10, 10, 10).astype(np.float32)
        sitk_image_5D = sitk.GetImageFromArray(array_5D)
        sitk.WriteImage(sitk_image_5D, file_path_5D)
        self.assertRaises(ValueError, utils.load_displacement, file_path_5D)
        file_path_5D.unlink()

        # load an image in wrong dtype
        file_path_wrong_dtype = Path(
            "registrationbaselines/tests/test_files/image_tmp.nii.gz")
        array_wrong_dtype = np.random.rand(
            3, 1, 10, 10, 10).astype(np.float64)
        sitk_image_wrong_dtype = sitk.GetImageFromArray(array_wrong_dtype)
        sitk.WriteImage(sitk_image_wrong_dtype, file_path_wrong_dtype)
        utils_nifti.set_intent_code(
            file_path_wrong_dtype, "NIFTI_INTENT_DISPVECT")

        self.assertRaises(TypeError, utils.load_displacement,
                          file_path_wrong_dtype)

        file_path_wrong_dtype.unlink()

    def test_save_image_interface(self):

        file_wrong_dim = torch.rand(10, 10, 10, 10)

        # should raise if path is not .nii or .nii.gz
        dummy_path = Path("dummy.txt")
        self.assertRaises(ValueError,
                          utils.save_image,
                          file_wrong_dim,
                          dummy_path,
                          (1, 1, 1, 1))

        # non-3D image should raise
        file_path_wrong_dim = Path(
            "registrationbaselines/tests/test_files/image_tmp.nii.gz")
        self.assertRaises(ValueError,
                          utils.save_image,
                          file_wrong_dim,
                          file_path_wrong_dim,
                          (1, 1, 1))

        # saving wotks
        file_path_correct = Path(
            "registrationbaselines/tests/test_files/image_tmp.nii.gz")
        file_correct = torch.rand(10, 10, 10)
        utils.save_image(file_correct, file_path_correct, (1, 1, 1))

        self.assertTrue(file_path_correct.exists())

        file_path_correct.unlink()
