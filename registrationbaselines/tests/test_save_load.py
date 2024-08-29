import unittest
from pathlib import Path
import sys

import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio

import registrationbaselines.warping.utils_displacement

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.core.utils as utils
import registrationbaselines.core.utils_nifti as utils_nifti


class TestSaveLoad(unittest.TestCase):

    def test_repeatedly_save_and_load_image(self) -> None:

        image_to_save = torch.rand(10, 10, 10)
        spacing = (1, 1, 1)
        file_path = Path("registrationbaselines/tests/test_files/tmp.nii.gz")

        for _ in range(10):

            utils.save_image(image_to_save, file_path, spacing)
            image_loaded = utils.load_image(file_path)

            self.assertAlmostEqual(image_to_save.detach().cpu().abs().sum().numpy(),
                                   image_loaded.detach().cpu().abs().sum().numpy(),
                                   delta=1e-4)
            self.assertAlmostEqual(image_to_save.detach().cpu().numpy().shape,
                                   image_loaded.detach().cpu().numpy().shape,
                                   delta=0.0)

            image_to_save = image_loaded

        file_path.unlink()

    def test_repeatedly_save_and_load_displacement(self) -> None:

        shape = (192, 128, 208, 3)
        spacing = (1.75, 1.75, 1.75)

        displacement_to_save = torch.rand(*shape)
        file_path = Path(
            "registrationbaselines/tests/test_files/tmp_disp.nii.gz")

        for _ in range(4):
            utils.save_displacement(displacement_to_save,
                                    file_path,
                                    spacing + (1,))

            displacement_loaded = utils.load_displacement(file_path)

            self.assertAlmostEqual(displacement_to_save.detach().cpu().abs().sum().numpy(),
                                   displacement_loaded.detach().cpu().abs().sum().numpy(),
                                   delta=1.1)
            self.assertAlmostEqual(displacement_to_save.detach().cpu().numpy().shape,
                                   displacement_loaded.detach().cpu().numpy().shape,
                                   delta=0.0)

            displacement_to_save = displacement_loaded

        file_path.unlink()

    def test_load_image_interface(self) -> None:

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

    def test_load_displacement_interface(self) -> None:

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

        # check that load_image returns a tensor
        file_path = Path(
            "registrationbaselines/tests/test_files/LungCT_0001_0001_deformation_to_LungCT_0001_0000.nii.gz")

        displacement = utils.load_displacement(file_path)
        self.assertTrue(isinstance(displacement, torch.Tensor))

        self.assertEqual(displacement.shape[-1], 3)
        self.assertEqual(len(displacement.shape), 4)

    def test_save_image_interface(self) -> None:

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

    def test_save_displacement_interface(self) -> None:

        # raises when not nifti
        dummy_path = Path("dummy.txt")
        self.assertRaises(ValueError,
                          utils.save_displacement,
                          torch.rand(10, 10, 10),
                          dummy_path,
                          (1, 1, 1))

        # raises when dim is not 4
        dummy_path = Path("dummy.nii.gz")
        image = torch.rand(10, 10, 10)
        self.assertRaises(ValueError,
                          utils.save_displacement,
                          image,
                          dummy_path,
                          (1, 1, 1))

        # raises when dim is 4, but wrong order
        image = torch.rand(10, 10, 10, 10)
        self.assertRaises(ValueError,
                          utils.save_displacement,
                          image,
                          dummy_path,
                          (1, 1, 1, 1))

        # raises when wrong dtype
        image = torch.rand(10, 10, 10, 3, dtype=torch.float64)
        self.assertRaises(TypeError,
                          utils.save_displacement,
                          image,
                          dummy_path,
                          (1, 1, 1, 1))

        # raise when wrong spacing dim
        image = torch.rand(10, 10, 10, 3, dtype=torch.float32)
        self.assertRaises(ValueError,
                          utils.save_displacement,
                          image,
                          dummy_path,
                          (1, 1, 1))

        # saved has right intent code
        image = torch.rand(10, 10, 10, 3, dtype=torch.float32)
        image = registrationbaselines.warping.utils_displacement.displacement_to_unit_displacement(
            image)

        utils.save_displacement(image, dummy_path, (1, 1, 1, 1))

        loaded = sitk.ReadImage(dummy_path)
        self.assertEqual("1006", loaded.GetMetaData("intent_code"))

        # saved exists
        self.assertTrue(dummy_path.exists())

        dummy_path.unlink()

    def test_load_keypoints(self) -> None:
        return

        # disabled for now because keypoint evaluation doesn't work yet

        # fails with non-existing path
        self.assertRaises(FileNotFoundError,
                          utils.load_keypoints, Path("dummy.txt"))

        # fails with 2D array with more than 3 columns
        array_2d = np.zeros((5, 5))
        path_2d = Path(
            "registrationbaselines/tests/test_files/keypoints_tmp.txt")
        np.savetxt(path_2d, array_2d, delimiter=',')

        self.assertRaises(ValueError, utils.load_keypoints, path_2d)
        path_2d.unlink()

        # switches x and y columns
        keypoints = np.random.rand(5, 3).astype(np.float32)
        path = Path(
            "registrationbaselines/tests/test_files/keypoints_tmp.txt")
        np.savetxt(path, keypoints, delimiter=',')

        loaded_keypoints = utils.load_keypoints(path)
        keypoints_torch = torch.from_numpy(keypoints)

        self.assertEqual(loaded_keypoints.shape, (5, 3))
        self.assertTrue(torch.equal(
            loaded_keypoints[:, 0], keypoints_torch[:, 2]))
        self.assertTrue(torch.equal(
            loaded_keypoints[:, 1], keypoints_torch[:, 1]))
        self.assertTrue(torch.equal(
            loaded_keypoints[:, 2], keypoints_torch[:, 0]))

        path.unlink()

    def test_load_torchio_vs_sitk(self) -> None:

        # loading with torchio vs loading with sitk and extracting np.array
        # results in a transpose of x and z

        path = Path(
            r"registrationbaselines/tests/test_files/LungCT_0001_0000_preprocessed_segmentation.nii.gz")

        image_array_tio = tio.ScalarImage(path).data.squeeze()

        image_array_sitk = sitk.GetArrayFromImage(sitk.ReadImage(path))

        shape_tio = tuple(image_array_tio.shape)
        shape_sitk = image_array_sitk.shape

        self.assertEqual(shape_tio[0], shape_sitk[2])
        self.assertEqual(shape_tio[1], shape_sitk[1])
        self.assertEqual(shape_tio[2], shape_sitk[0])

        self.assertNotEqual(shape_tio, shape_sitk)
