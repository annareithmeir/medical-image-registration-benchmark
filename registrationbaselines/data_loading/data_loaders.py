import glob
import random
from itertools import combinations
from pathlib import Path
from typing import List, Union, Tuple
import shutil

from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import SimpleITK as sitk

from . import utils as dataloader_utils
import registrationbaselines.core.utils as utils
from registrationbaselines.core.types import datasetReturnType, floatArray2D

# global clipping [min,max] values
# todo verify clipping parameters
WINDOW_BONES = [-400, 1600]
WINDOW_SOFT_TISSUE = [-150, 250]

"""
Parent class datasets
"""


class GenericDataset(Dataset[datasetReturnType]):

    def __init__(self, name: str, return_type: str = None, indices: list[int] = None, **kwargs) -> None:
        super().__init__()

        self.images_path = None
        self.images_path_preprocessed = None
        self.indices = indices
        self.ndim = None
        self.spacing: Tuple[int, ...]
        self.image_shape = None
        self.return_type = return_type
        if return_type is None:
            self.return_type = "torch_tensor_dict"
        self.images_list = list()

        self.has_segmentations = False
        self.has_keypoints = False

        self.name = name

        assert self.return_type in ["torch_tensor_dict", "path_dict"]

    def __len__(self):
        return len(self.images_list)

    def _get_image_pair_as_tensors(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image_fixed = utils.load_image(
            self.images_path / self.images_list[idx][0])
        image_moving = utils.load_image(
            self.images_path / self.images_list[idx][1])

        return image_fixed, image_moving

    def _get_image_pair_as_paths(self, idx):
        return self.images_path / self.images_list[idx][0], self.images_path / self.images_list[idx][1]

    def _get_segmentation_pair_as_tensors(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        segmentation_fixed = utils.load_image(
            self.images_path / self.segmentations_list[idx][0])
        segmentation_moving = utils.load_image(
            self.images_path / self.segmentations_list[idx][1])

        return segmentation_fixed, segmentation_moving

    def _get_segmentation_pair_as_paths(self, idx):
        return self.images_path / self.segmentations_list[idx][0], self.images_path / self.segmentations_list[idx][1]

    def _get_keypoint_pair_as_tensors(self, idx):
        pass

    def _get_keypoint_pair_as_paths(self, idx):
        pass

    def __getitem__(self, idx):
        if (self.has_keypoints is False) and (self.has_segmentations is False):
            if self.return_type == "path_dict":
                fixed_image, moving_image = self._get_image_pair_as_paths(idx)
                item = {"moving_image": moving_image,
                        "fixed_image": fixed_image}
            elif self.return_type == "torch_tensor_dict":  # np_array bsxhxw
                fixed_image, moving_image = self._get_image_pair_as_tensors(
                    idx)
                item = {"moving_image": moving_image,
                        "fixed_image": fixed_image}

        elif (self.has_keypoints is False) and (self.has_segmentations is True):
            if self.return_type == "path_dict":
                path_f, path_m = self._get_image_pair_as_paths(idx)
                path_segmentations_f, path_segmentations_m = self._get_segmentation_pair_as_paths(
                    idx)
                item = {"fixed_image": path_f, "moving_image": path_m,
                        "fixed_segmentations": path_segmentations_f, "moving_segmentations": path_segmentations_m}
            elif self.return_type == "torch_tensor_dict":  # np_array bsxhxw
                fixed_image, moving_image = self._get_image_pair_as_tensors(
                    idx)
                segmentations_f, segmentations_m = self._get_segmentation_pair_as_tensors(
                    idx)
                item = {"moving_image": moving_image, "fixed_image": fixed_image,
                        "moving_segmentations": segmentations_m, "fixed_segmentations": segmentations_f}

        elif (self.has_keypoints is True) and (self.has_segmentations is False):
            if self.return_type == "path_dict":
                path_f, path_m = self._get_image_pair_as_paths(idx)
                path_keypoints_f, path_keypoints_m = self._get_keypoint_pair_as_paths(
                    idx)
                item = {"fixed_image": path_f, "moving_image": path_m, "fixed_keypoints": path_keypoints_f,
                        "moving_keypoints": path_keypoints_m}
            elif self.return_type == "torch_tensor_dict":  # np_array bsxhxw
                fixed_image, moving_image = self._get_image_pair_as_tensors(
                    idx)
                keypoints_f, keypoints_m = self._get_keypoint_pair_as_tensors(
                    idx)
                item = {"moving_image": moving_image, "fixed_image": fixed_image, "moving_keypoints": keypoints_m,
                        "fixed_keypoints": keypoints_f}

        elif (self.has_keypoints is True) and (self.has_segmentations is True):
            if self.return_type == "path_dict":
                path_f, path_m = self._get_image_pair_as_paths(idx)
                path_segmentations_f, path_segmentations_m = self._get_segmentation_pair_as_paths(
                    idx)
                path_keypoint_f, path_keypoint_m = self._get_keypoint_pair_as_paths(
                    idx)
                item = {"fixed_image": path_f, "moving_image": path_m, "fixed_segmentations": path_segmentations_f,
                        "moving_segmentations": path_segmentations_m, "fixed_keypoints": path_keypoint_f, "moving_keypoints": path_keypoint_m}
            elif self.return_type == "torch_tensor_dict":  # np_array bsxhxw
                fixed_image, moving_image = self._get_image_pair_as_tensors(
                    idx)
                segmentations_f, segmentations_m = self._get_segmentation_pair_as_tensors(
                    idx)
                keypoint_f, keypoint_m = self._get_keypoint_pair_as_tensors(
                    idx)
                item = {"moving_image": moving_image, "fixed_image": fixed_image, "moving_segmentations": segmentations_m,
                        "fixed_segmentations": segmentations_f, "fixed_keypoints": keypoint_f, "moving_keypoints": keypoint_m}
        return item

    def plot_random_image(self) -> None:
        """
        Plots a random image of the dataset including segmentations and keypoints
        """

        rand_idx = random.randint(0, len(self) - 1)

        fixed_image, moving_image = self._get_image_pair_as_tensors(rand_idx)
        if self.has_segmentations:
            segmentation_f, segmentation_m = self._get_segmentation_pair_as_tensors(
                rand_idx)
        if self.has_keypoints:
            keypoints_f, keypoints_m = self._get_keypoint_pair_as_tensors(
                rand_idx)
        fixed_image = fixed_image.numpy()
        moving_image = moving_image.numpy()

        fig = plt.figure(figsize=(20, 12))

        if self.ndim == 2:
            # moving image
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(moving_image, cmap='gray')
            if self.has_segmentations:
                plt.imshow(segmentation_m, alpha=0.3)
            if self.has_keypoints:
                plt.scatter(keypoints_m[:, 0],
                            keypoints_m[:, 1], marker='x', c='red')
            plt.colorbar()
            plt.title("Moving")

            # fixed image
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(fixed_image, cmap='gray')
            if self.has_segmentations:
                plt.imshow(segmentation_f, alpha=0.3)
            if self.has_keypoints:
                plt.scatter(keypoints_f[:, 0],
                            keypoints_f[:, 1], marker='x', c='red')
            plt.colorbar()
            plt.title("Fixed")
        else:
            image_size = self.image_shape
            slices = [int(image_size[0] / 2), int(image_size[1] / 2),
                      int(image_size[2] / 2)]

            for a in range(0, 3):
                # moving image
                ax = fig.add_subplot(2, 3, a + 1)
                if a == 0:
                    slice_moving_image = moving_image[slices[a], :, :]
                    if self.has_segmentations:
                        slice_segmentation_m = segmentation_m[slices[a], :, :]
                    if self.has_keypoints:
                        slice_keypoints_m = keypoints_m[np.where(
                            abs(keypoints_m[:, 0] - slices[a]) <= 0.5)]
                        slice_keypoints_m = slice_keypoints_m[:, 1:]
                if a == 1:
                    slice_moving_image = moving_image[:, slices[a], :]
                    if self.has_segmentations:
                        slice_segmentation_m = segmentation_m[:, slices[a], :]
                    if self.has_keypoints:
                        slice_keypoints_m = keypoints_m[np.where(
                            abs(keypoints_m[:, 1] - slices[a]) <= 0.5)]
                        slice_keypoints_m = slice_keypoints_m[:, [0, 2]]
                if a == 2:
                    slice_moving_image = moving_image[:, :, slices[a]]
                    if self.has_segmentations:
                        slice_segmentation_m = segmentation_m[:, :, slices[a]]
                    if self.has_keypoints:
                        slice_keypoints_m = keypoints_m[np.where(
                            abs(keypoints_m[:, 2] - slices[a]) <= 0.5)]
                        slice_keypoints_m = slice_keypoints_m[:, :-1]

                plt.imshow(slice_moving_image, cmap='gray')
                plt.colorbar()
                if self.has_segmentations:
                    plt.imshow(slice_segmentation_m, alpha=0.3)
                if self.has_keypoints:
                    plt.scatter(
                        slice_keypoints_m[:, 1], slice_keypoints_m[:, 0], marker='x', c='red')
                # plt.gca().invert_yaxis()

                # fixed image
                ax = fig.add_subplot(2, 3, a + 4)
                if a == 0:
                    slice_fixed_image = fixed_image[slices[a], :, :]
                    if self.has_segmentations:
                        slice_segmentation_f = segmentation_f[slices[a], :, :]
                    if self.has_keypoints:
                        slice_keypoints_f = keypoints_f[np.where(
                            abs(keypoints_f[:, 0] - slices[a]) <= 0.5)]
                        slice_keypoints_f = slice_keypoints_f[:, 1:]
                if a == 1:
                    slice_fixed_image = fixed_image[:, slices[a], :]
                    if self.has_segmentations:
                        slice_segmentation_f = segmentation_f[:, slices[a], :]
                    if self.has_keypoints:
                        slice_keypoints_f = keypoints_f[np.where(
                            abs(keypoints_f[:, 1] - slices[a]) <= 0.5)]
                        slice_keypoints_f = slice_keypoints_f[:, [0, 2]]
                if a == 2:
                    slice_fixed_image = fixed_image[:, :, slices[a]]
                    if self.has_segmentations:
                        slice_segmentation_f = segmentation_f[:, :, slices[a]]
                    if self.has_keypoints:
                        slice_keypoints_f = keypoints_f[np.where(
                            abs(keypoints_f[:, 2] - slices[a]) <= 0.5)]
                        slice_keypoints_f = slice_keypoints_f[:, :-1]

                plt.imshow(slice_fixed_image, cmap='gray')
                plt.colorbar()
                if self.has_segmentations:
                    plt.imshow(slice_segmentation_f, alpha=0.3)
                if self.has_keypoints:
                    plt.scatter(
                        slice_keypoints_f[:, 1], slice_keypoints_f[:, 0], marker='x', c='red')

        plt.tight_layout()
        plt.suptitle("idx: {}".format(rand_idx))
        plt.show()


"""
Toy datasets
"""


class MNISTDataset(GenericDataset):
    """
    Returns image pairs of the same digit of the MNSIt dataset.
    The images are of shape (32,32) and normalized to [0,1]
    """

    def __init__(self, num_pairs: int, return_type: str = None):
        """

        @param num_pairs: Amount of image pairs to use from the overall dataset
        @param return_type: in what data format the images should be returned in __getitem__()
        """

        super().__init__("MNIST", return_type, None)

        assert return_type == "torch_tensor"  # path is not applicable for MNIST dataset

        self.ndim = 2
        self.spacing = (1, 1)
        self.image_shape = (32, 32)  # vmx needs 2N, N=number of layers
        self.num_pairs = num_pairs
        self.has_segmentations = False
        self.has_keypoints = False

        # Define transformations for the dataset including the custom transform
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])

        # Download and load the training dataset
        self.dataset = datasets.MNIST(
            root='./data', train=False, download=True)

        # Create a dictionary to store indices of each digit
        self.digit_indices = {i: [] for i in range(10)}
        for idx, (image, segmentation) in enumerate(self.dataset):
            self.digit_indices[segmentation].append(idx)

        self._read_image_idx_pairs()

    def _read_image_idx_pairs(self) -> None:
        """
        Extracts image pairs of similar indices from the data
        @return:
        """
        for idx in range(self.num_pairs):
            image1, segmentation = self.dataset[idx]
            # Randomly select another image with the same segmentation
            image2_idx = random.choice(self.digit_indices[segmentation])
            self.images_list.append([idx, image2_idx])

    def _get_image_pair_as_tensors(self, idx: int):
        """
        Returns the image pair at index idx. We normalize and reshape the images here.
        @param idx: Index of desired image pair
        @return: fixed image, moving image of shape (32,32)
        """

        fixed_image, _ = self.dataset[self.images_list[idx][0]]
        moving_image, _ = self.dataset[self.images_list[idx][1]]

        fixed_image = self.transforms(fixed_image)
        moving_image = self.transforms(moving_image)

        moving_image = dataloader_utils.normalize_tensor_to_0_1(moving_image)
        fixed_image = dataloader_utils.normalize_tensor_to_0_1(fixed_image)

        return fixed_image.squeeze(), moving_image.squeeze()


"""
Medical datasets
"""


class L2RLungCTDataset(GenericDataset):
    """
    Learn2Reg Lung CT dataset (available at https://learn2reg.grand-challenge.org/Datasets/)
    Intra-patient
    moving image: full, fixedimage: cropped
    Currently only using the 20 training image pairs since the test data has no annotations.
    We assume the data is preprocessed with preprocess() before use
    """

    def __init__(self, dataset_path: Path, return_type: str = None, indices: list[int] = None) -> None:
        """

        @param dataset_path: Path to the original or pre-processed dataset
        @param transforms: transformations for the pre-processing, if desired
        @param return_type: The data can either be returned as a dict[Path], dict[np.ndarray] or tuple of np.ndarray
        @param indices: If desired, only specific indices can be used for the dataset creation (e.g. for train/val/test split)
        """

        super().__init__("LungCT", return_type, indices)

        self.images_path = dataset_path
        self.images_path_preprocessed = None
        self.ndim = 3

        self.has_segmentations = True
        self.has_keypoints = True

        self.segmentation_segmentations = {
            0: "background",
            1: "lung"
        }

        self.images_list = None
        self.segmentations_list = None
        self.keypoints_list = None
        self._load_images_list()
        self._load_segmentations_list()
        self._load_keypoints_list()
        if indices is not None:  # create subsets for e.g. validation and training
            self.images_list = [self.images_list[i] for i in indices]
            self.segmentations_list = [
                self.segmentations_list[i] for i in indices]
            self.keypoints_list = [self.keypoints_list[i] for i in indices]

    def _get_keypoint_pair_as_tensors(self, idx):
        keypoints_f = utils.load_keypoints(
            self.images_path / self.keypoints_list[idx][0])

        keypoints_m = utils.load_keypoints(
            self.images_path / self.keypoints_list[idx][1])
        return keypoints_f, keypoints_m

    def _get_keypoint_pair_as_paths(self, idx) -> Tuple[Path, Path]:
        return self.images_path / self.keypoints_list[idx][0], self.images_path / self.keypoints_list[idx][1]

    def preprocess(self, save_path: Path) -> None:
        """
        Preprocessing of the whole dataset
        @param save_path: The path where the preprocessing data should be saved. The same folder structure as in the
         original dataset will be created there automatically and after preprocessing, the data will be loaded from this path instead of the original one.
        @return: None
        """

        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (save_path / "masksTr").mkdir(parents=True, exist_ok=True)
        (save_path / "keypointsTr").mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(self)), desc="Preprocessing", unit="iteration"):
            file_moving_image = self.images_list[idx][0]
            file_fixed_image = self.images_list[idx][1]
            file_segmentation_m = self.segmentations_list[idx][0]
            file_segmentation_f = self.segmentations_list[idx][1]
            file_keypoints_m = self.keypoints_list[idx][0]
            file_keypoints_f = self.keypoints_list[idx][1]

            subject_dict = {
                "moving_image": tio.ScalarImage(self.images_path / self.images_list[idx][0]),
                "fixed_image": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
                "segmentation_m": tio.LabelMap(self.images_path / self.segmentations_list[idx][0]),
                "segmentation_f": tio.LabelMap(self.images_path / self.segmentations_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            keypoints_m = np.genfromtxt(
                self.images_path / self.keypoints_list[idx][0], delimiter=',')
            keypoints_f = np.genfromtxt(
                self.images_path / self.keypoints_list[idx][1], delimiter=',')

            # clip bones
            clip = tio.Clamp(out_min=WINDOW_BONES[0], out_max=WINDOW_BONES[1])
            subject = clip(subject)

            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["moving_image"].numpy().min(), subject["moving_image"].numpy().max()))
            rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["fixed_image"].numpy().min(), subject["fixed_image"].numpy().max()))

            subject["moving_image"] = rescale_x(subject["moving_image"])
            subject["fixed_image"] = rescale_y(subject["fixed_image"])

            resample = tio.Resample(1.75)
            subject = resample(subject)
            self.image_shape = subject["moving_image"].data.shape[1:]
            self.spacing = (1.75, 1.75, 1.75)

            # after resmpling, the keypoints coordinates need to be adapted
            keypoints_m[:, 1] = keypoints_m[:, 1] * 1.25 / 1.75
            keypoints_f[:, 1] = keypoints_f[:, 1] * 1.25 / 1.75

            # save preprocessed images
            subject["moving_image"].save(save_path / file_moving_image)
            subject["fixed_image"].save(save_path / file_fixed_image)
            subject["segmentation_m"].save(save_path / file_segmentation_m)
            subject["segmentation_f"].save(save_path / file_segmentation_f)
            np.savetxt(save_path / file_keypoints_m,
                       keypoints_m, delimiter=",")
            np.savetxt(save_path / file_keypoints_f,
                       keypoints_f, delimiter=",")

        self.images_path = save_path
        print("From now on reading images from ", self.images_path)

    def _load_images_list(self) -> None:
        """
        Initializes the image list from the given dataset path and indices
        @return:
        """

        self.images_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = Path("imagesTr/" + file_str + "_0001.nii.gz")
            file_f = Path("imagesTr/" + file_str + "_0000.nii.gz")
            self.images_list.append([file_f, file_m])

        import SimpleITK as sitk
        self.spacing = sitk.ReadImage(
            self.images_path / self.images_list[0][0]).GetSpacing()
        self.image_shape = sitk.GetArrayFromImage(
            sitk.ReadImage(self.images_path / self.images_list[0][0])).shape

    def _load_segmentations_list(self) -> None:
        """
        Initializes the segmentations list from the given dataset path and indices
        @return:
        """
        self.segmentations_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = Path("masksTr/" + file_str + "_0001.nii.gz")
            file_f = Path("masksTr/" + file_str + "_0000.nii.gz")
            self.segmentations_list.append([file_f, file_m])

    def _load_keypoints_list(self) -> None:
        """
        Initializes the keypoints list from the given dataset path and indices
        @return:
        """
        self.keypoints_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = Path("keypointsTr/" + file_str + "_0001.csv")
            file_f = Path("keypointsTr/" + file_str + "_0000.csv")
            self.keypoints_list.append([file_f, file_m])


class L2RAbdominalMRCTDataset(GenericDataset):
    """
    Learn2Reg Abdominal MR/CT dataset (available at https://learn2reg.grand-challenge.org/Datasets/)
    Intra-patient
    moving: CT, fixed: MR
    Currently only using the 8 paired and segmentationed images.
    The original dataset contains 8 in imagesTr and 8 in imagesTs. Only 8 in Ts are segmentationed.
    preprocess() optional, since already isotropic pixel size
    """

    def __init__(self, dataset_path: Path, return_type: str = None,
                 indices: list[int] = None) -> None:
        """

        @param dataset_path: path to the original or preprocessed dataset
        @param transforms: The preprocessing steps to be done (normalize, clip, resize)
        @param return_type: data type of return values. Either dict of paths, dict of np arrays or np arrays
        @param indices: list of indices which form the dataset (e.g. for train/val/test split)
        """

        super().__init__("AbdomenMRCT", return_type, indices)

        self.images_path = dataset_path
        self.images_path_preprocessed = None
        self.ndim = 3

        self.has_segmentations = True
        self.has_keypoints = False

        self.segmentation_segmentations = {
            0: "background",
            1: "liver",
            2: "spleen",
            3: "right kidney",
            4: "left kidney"
        }

        self.images_list = None
        self.segmentations_list = None
        self._load_images_list()
        self._load_segmentations_list()
        if indices is not None:  # create subsets for e.g. validation and training
            self.images_list = [self.images_list[i] for i in indices]
            self.segmentations_list = [
                self.segmentations_list[i] for i in indices]

    def preprocess(self, save_path: Path) -> None:
        """
        Preprocessing of the whole dataset
        @param save_path: The path where the preprocessing data should be saved. The same folder structure as in the
         original dataset will be created there automatically and after preprocessing, the data will be loaded from this path instead of the original one.
        @return: None
        """

        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (save_path / "labelsTr").mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(self)), desc="Preprocessing", unit="iteration"):
            file_moving_image = self.images_list[idx][1]
            file_fixed_image = self.images_list[idx][0]
            file_segmentation_m = self.segmentations_list[idx][1]
            file_segmentation_f = self.segmentations_list[idx][0]

            subject_dict = {
                "fixed_image": tio.ScalarImage(self.images_path / self.images_list[idx][0]),
                "moving_image": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
                "segmentation_f": tio.LabelMap(self.images_path / self.segmentations_list[idx][0]),
                "segmentation_m": tio.LabelMap(self.images_path / self.segmentations_list[idx][1])
            }
            subject = tio.Subject(subject_dict)

            # clip = tio.Clamp(
            #     out_min=WINDOW_BONES[0], out_max=WINDOW_BONES[1])
            clip = tio.Clamp(
                out_min=WINDOW_SOFT_TISSUE[0], out_max=WINDOW_SOFT_TISSUE[1])
            subject["moving_image"] = clip(subject["moving_image"])

            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["moving_image"].numpy().min(), subject["moving_image"].numpy().max()))
            rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["fixed_image"].numpy().min(), subject["fixed_image"].numpy().max()))

            subject["moving_image"] = rescale_x(subject["moving_image"])
            subject["fixed_image"] = rescale_y(subject["fixed_image"])

            # save preprocessed images
            subject["moving_image"].save(save_path / file_moving_image)
            subject["fixed_image"].save(save_path / file_fixed_image)
            subject["segmentation_m"].save(save_path / file_segmentation_m)
            subject["segmentation_f"].save(save_path / file_segmentation_f)

        self.images_path = save_path
        print("From now on reading images from ", self.images_path)

    def _load_images_list(self) -> None:
        """
        Initializes the image list from the given dataset path and indices
        @return:
        """

        self.images_list = list()

        for i in range(1, 9):
            file_str = "AbdomenMRCT_" + str(i).zfill(4)
            file_m = "imagesTr/" + file_str + "_0001.nii.gz"
            file_f = "imagesTr/" + file_str + "_0000.nii.gz"
            self.images_list.append([file_f, file_m])

        import SimpleITK as sitk
        self.spacing = sitk.ReadImage(
            self.images_path / self.images_list[0][0]).GetSpacing()
        self.image_shape = sitk.GetArrayFromImage(
            sitk.ReadImage(self.images_path / self.images_list[0][0])).shape

    def _load_segmentations_list(self) -> None:
        """
        Initializes the segmentations list from the given dataset path and indices
        @return:
        """

        self.segmentations_list = list()

        for i in range(1, 9):
            file_str = "AbdomenMRCT_" + str(i).zfill(4)
            file_m = "labelsTr/" + file_str + "_0001.nii.gz"
            file_f = "labelsTr/" + file_str + "_0000.nii.gz"
            self.segmentations_list.append([file_f, file_m])


class L2RAbdominalCTCTDataset(GenericDataset):
    """
    Learn2Reg Abdominal CT dataset (available at https://learn2reg.grand-challenge.org/Datasets/)
    Inter-patient
    Images are of shape (192, 160, 256)
    Currently only using the 30 segmentationed images and all possible combinations among them (435 pairs)
    preprocess() optional since already isotropic pixel size
    """

    def __init__(self, dataset_path: Path, return_type: str = None,
                 indices: list[int] = None) -> None:
        """

        @param dataset_path: path to the original or preprocessed dataset
        @param transforms: The preprocessing steps to be done (normalize, clip, resize)
        @param return_type: data type of return values. Either dict of paths, dict of np arrays or np arrays
        @param indices: list of indices which form the dataset (e.g. for train/val/test split)
        """

        super().__init__("AbdomenCTCT", return_type, indices)

        self.images_path = dataset_path
        self.images_path_preprocessed = None
        self.ndim = 3

        self.has_segmentations = True
        self.has_keypoints = False

        self.classes = {0: "background",
                        1: "spleen",
                        2: "right kidney",
                        3: "left kidney",
                        4: "gall bladder",
                        5: "esophagus",
                        6: "liver",
                        7: "stomach",
                        8: "aorta",
                        9: "inferior vena cava",
                        10: "portal and splenic vein",
                        11: "pancreas",
                        12: "left adrenal gland",
                        13: "right adrenal gland"}

        self.images_list = None
        self.segmentations_list = None
        self._load_images_list()
        self._load_segmentations_list()
        if indices is not None:  # create subsets for e.g. validation and training
            self.images_list = [self.images_list[i] for i in indices]
            self.segmentations_list = [
                self.segmentations_list[i] for i in indices]

    def preprocess(self, save_path: Path) -> None:
        """
        Preprocessing of the whole dataset
        @param save_path: The path where the preprocessing data should be saved. The same folder structure as in the
         original dataset will be created there automatically and after preprocessing, the data will be loaded from this path instead of the original one.
        @return: None
        """

        files_images = list()
        files_segmentations = list()
        for i in range(1, 31):
            file_str = "AbdomenCTCT_" + str(i).zfill(4)
            file_seg = "labelsTr/" + file_str + "_0000.nii.gz"
            file_img = "imagesTr/" + file_str + "_0000.nii.gz"
            files_segmentations.append(file_seg)
            files_images.append(file_img)

        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (save_path / "labelsTr").mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(files_images)), desc="Preprocessing",
                        unit="iteration"):
            # since inter-patient, pairs not given
            file_moving_image = files_images[idx]
            file_segmentation_m = files_segmentations[idx]

            subject_dict = {
                "moving_image": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
                "segmentation_m": tio.LabelMap(self.images_path / self.segmentations_list[idx][1])
            }
            subject = tio.Subject(subject_dict)

            # clip = tio.Clamp(
            #     out_min=WINDOW_BONES[0], out_max=WINDOW_BONES[1])
            clip = tio.Clamp(
                out_min=WINDOW_SOFT_TISSUE[0], out_max=WINDOW_SOFT_TISSUE[1])
            subject = clip(subject)

            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["moving_image"].numpy().min(), subject["moving_image"].numpy().max()))

            subject["moving_image"] = rescale_x(subject["moving_image"])

            # save preprocessed images
            subject["moving_image"].save(save_path / file_moving_image)
            subject["segmentation_m"].save(save_path / file_segmentation_m)

        self.images_path = save_path
        print("From now on reading images from ", self.images_path)

    def _load_images_list(self) -> None:
        """
        Initializes the image list from the given dataset path and indices
        @return:
        """

        self.images_list = list()
        files = list()
        for i in range(1, 31):
            file_str = "AbdomenCTCT_" + str(i).zfill(4)
            file = "imagesTr/" + file_str + "_0000.nii.gz"
            files.append(file)

        self.images_list = [(x, y)
                            for x, y in combinations(files, 2) if x != y]

        import SimpleITK as sitk
        self.spacing = sitk.ReadImage(
            self.images_path / self.images_list[0][0]).GetSpacing()
        self.image_shape = sitk.GetArrayFromImage(
            sitk.ReadImage(self.images_path / self.images_list[0][0])).shape

    def _load_segmentations_list(self) -> None:
        """
        Initializes the segmentations list from the given dataset path and indices
        @return:
        """

        self.segmentations_list = list()
        files = list()
        for i in range(1, 31):
            file_str = "AbdomenCTCT_" + str(i).zfill(4)
            file = "labelsTr/" + file_str + "_0000.nii.gz"
            files.append(file)

        self.segmentations_list = [(x, y)
                                   for x, y in combinations(files, 2) if x != y]


class ACDCDataset(GenericDataset):
    """
    source: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb
    Number of subjects: 150 (100 training, rest split in 25 test and 25 val)
    modality: MR
    anatomy: cardiac
    m,f: fixed=ed/01, moving=es/1x

    we assume that the images have been preprocessed to 2d-middle slices, shape (128, 128), normalized and resampled to isotropic 1.8 (can be done with preprocess())
    """

    def __init__(self, dataset_path: Path, return_type: str = None, return_mode: str = "train", indices: list[int] = None) -> None:
        """

        @param dataset_path:
        @param return_type:
        @param return_mode: train/val/test subdataset
        @param indices: unused
        """

        super().__init__("ACDC", return_type, indices)

        self.spacing = (1.8, 1.8)
        self.image_shape = (128, 128)
        self.ndim = 2

        self.has_segmentations = True
        self.has_keypoints = False

        self.images_path = dataset_path
        assert return_mode in ["train", "val", "test"]
        self.return_mode = return_mode

        self.classes = {
            0: "background",
            1: "RV",  # right ventricle
            2: "LV-Myo",  # epicardium
            3: "LV-BP"  # endocardium
        }

        self.group_map = {
            "NOR": 0,
            "MINF": 1,
            "DCM": 2,
            "HCM": 3,
            "RV": 4
        }

        self.images_list, self.segmentations_list, self.groups = self.read_filenames()

        if indices is not None:
            self.images_list = [self.images_list[i] for i in indices]
            self.segmentations_list = [
                self.segmentations_list[i] for i in indices]
            self.groups = [self.groups[i] for i in indices]

    def read_filenames(self):
        x_ls = list()
        y_ls = list()
        masks_x_ls = list()
        masks_y_ls = list()
        group_ls = list()

        if "train" in self.return_mode:
            # load train paths. 01=fixed, 1x=moving
            for i in range(1, 101):
                file_str = "patient" + str(i).zfill(3)
                file_str_full = self.images_path / \
                    ('training/' + file_str + "/" +
                     file_str + '_frame*_gt.nii.gz')
                ids = sorted(glob.glob(str(file_str_full)))
                m_id = ids[1][-12:-10]
                f_id = ids[0][-12:-10]
                y_ls.append(
                    self.images_path / ("training/" + file_str + "/" + file_str + "_frame" + f_id + ".nii.gz"))
                masks_y_ls.append(
                    self.images_path / ("training/" + file_str + "/" + file_str + "_frame" + f_id + "_gt.nii.gz"))
                x_ls.append(
                    self.images_path / ("training/" + file_str + "/" + file_str + "_frame" + m_id + ".nii.gz"))
                masks_x_ls.append(
                    self.images_path / ("training/" + file_str + "/" + file_str + "_frame" + m_id + "_gt.nii.gz"))

                with open(self.images_path / ("training/" + file_str + "/Info.cfg"), 'r') as file:
                    for i, line in enumerate(file):
                        if i == 2:
                            group = line.split(':')[1].strip()
                            break

                group_ls.append(self.group_map[group])

        elif "val" in self.return_mode:
            for i in range(101, 126):
                file_str = "patient" + str(i).zfill(3)
                file_str_full = self.images_path / \
                    ("testing/" + file_str + "/" +
                     file_str + "_frame*_gt.nii.gz")
                ids = sorted(glob.glob(str(file_str_full)))
                m_id = ids[1][-12:-10]
                f_id = ids[0][-12:-10]
                y_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + f_id + ".nii.gz"))
                masks_y_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + f_id + "_gt.nii.gz"))
                x_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + m_id + ".nii.gz"))
                masks_x_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + m_id + "_gt.nii.gz"))

                with open(self.images_path / ("testing/" + file_str + "/Info.cfg"), 'r') as file:
                    for i, line in enumerate(file):
                        if i == 2:
                            group = line.split(':')[1].strip()
                            break

                group_ls.append(self.group_map[group])
        else:  # test
            for i in range(126, 151):
                file_str = "patient" + str(i).zfill(3)
                file_str_full = self.images_path / \
                    ("testing/" + file_str + "/" +
                     file_str + "_frame*_gt.nii.gz")
                ids = sorted(glob.glob(str(file_str_full)))
                m_id = ids[1][-12:-10]
                f_id = ids[0][-12:-10]
                y_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + f_id + ".nii.gz"))
                masks_y_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + f_id + "_gt.nii.gz"))
                x_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + m_id + ".nii.gz"))
                masks_x_ls.append(
                    self.images_path / ("testing/" + file_str + "/" + file_str + "_frame" + m_id + "_gt.nii.gz"))
                with open(self.images_path / ("testing/" + file_str + "/Info.cfg"), 'r') as file:
                    for i, line in enumerate(file):
                        if i == 2:
                            group = line.split(':')[1].strip()
                            break

                group_ls.append(self.group_map[group])

        return list(zip(x_ls, y_ls)), list(zip(masks_x_ls, masks_y_ls)), group_ls

    def preprocess(self, save_path: Path) -> None:
        """
        Preprocesses the dataset to uniform pixel size, cropping around heart, 2d middle slice
        @param save_path: path to save the preprocessed images to
        @return:
        """

        if "train" in self.return_mode:
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "training").mkdir(parents=True, exist_ok=True)
            (save_path / "training").mkdir(parents=True, exist_ok=True)
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "testing").mkdir(parents=True, exist_ok=True)
            (save_path / "testing").mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(self)), desc="Preprocessing", unit="iteration"):
            file_moving_image = self.images_list[idx][1]
            file_fixed_image = self.images_list[idx][0]
            file_segmentation_m = self.segmentations_list[idx][1]
            file_segmentation_f = self.segmentations_list[idx][0]

            subject_dict = {
                "fixed_image": tio.ScalarImage(self.images_path / self.images_list[idx][0]),
                "moving_image": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
                "segmentation_f": tio.ScalarImage(self.images_path / self.segmentations_list[idx][0]),
                "segmentation_m": tio.ScalarImage(self.images_path / self.segmentations_list[idx][1])
            }
            subject = tio.Subject(subject_dict)
            resample_uniform = tio.Resample(1.8)
            subject = resample_uniform(subject)

            crop_roi = tio.CropOrPad(
                (128, 128, 128), mask_name="segmentations_y")
            subject = crop_roi(subject)

            slice = subject["moving_image"].numpy().shape[-1] // 2
            segmentations_x_tmp = subject["segmentation_m"].numpy(
            )[..., slice, np.newaxis]
            segmentations_y_tmp = subject["segmentation_f"].numpy(
            )[..., slice, np.newaxis]
            if len(np.unique(segmentations_x_tmp)) < 4 or len(np.unique(segmentations_y_tmp)) < 4:
                print("not all segmentations present in slice.")

            subject["moving_image"] = tio.ScalarImage(
                tensor=subject["moving_image"].numpy()[..., slice, np.newaxis])
            subject["fixed_image"] = tio.ScalarImage(
                tensor=subject["fixed_image"].numpy()[..., slice, np.newaxis])
            subject["segmentation_m"] = tio.ScalarImage(
                tensor=segmentations_x_tmp)
            subject["segmentation_f"] = tio.ScalarImage(
                tensor=segmentations_y_tmp)

            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["moving_image"].numpy().min(), subject["moving_image"].numpy().max()))
            rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["fixed_image"].numpy().min(), subject["fixed_image"].numpy().max()))

            subject["moving_image"] = rescale_x(subject["moving_image"])
            subject["fixed_image"] = rescale_y(subject["fixed_image"])

            # rescale = tio.RescaleIntensity(out_min_max=(0, 1))
            # subject = rescale(subject)

            # save preprocessed images
            if "train" in self.return_mode:
                save_path.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.images_path / "training" / file_moving_image.parent.name /
                                "Info.cfg", save_path / "training" / file_moving_image.parent.name / "Info.cfg")
                (save_path / "training" /
                 file_moving_image.parent.name).mkdir(parents=True, exist_ok=True)
                (save_path / "training" /
                 file_moving_image.parent.name).mkdir(parents=True, exist_ok=True)
                subject["moving_image"].save(
                    save_path / "training" / file_moving_image.parent.name / file_moving_image.name)
                subject["fixed_image"].save(
                    save_path / "training" / file_fixed_image.parent.name / file_fixed_image.name)
                subject["segmentation_m"].save(
                    save_path / "training" / file_segmentation_m.parent.name / file_segmentation_m.name)
                subject["segmentation_f"].save(
                    save_path / "training" / file_segmentation_f.parent.name / file_segmentation_f.name)
            else:

                save_path.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.images_path / "testing" / file_moving_image.parent.name / "Info.cfg",
                                save_path / "testing" / file_moving_image.parent.name / "Info.cfg")
                (save_path / "testing" /
                 file_moving_image.parent.name).mkdir(parents=True, exist_ok=True)
                (save_path / "testing" /
                 file_moving_image.parent.name).mkdir(parents=True, exist_ok=True)
                subject["moving_image"].save(
                    save_path / "testing" / file_moving_image.parent.name / file_moving_image.name)
                subject["fixed_image"].save(
                    save_path / "testing" / file_fixed_image.parent.name / file_fixed_image.name)
                subject["segmentation_m"].save(
                    save_path / "testing" / file_segmentation_m.parent.name / file_segmentation_m.name)
                subject["segmentation_f"].save(
                    save_path / "testing" / file_segmentation_f.parent.name / file_segmentation_f.name)

        self.images_path = save_path
        print("From now on reading images from ", self.images_path)


class FIREDataset(GenericDataset):
    """
    Retina image dataset available at https://projects.ics.forth.gr/cvrl/fire/
    134 retina image pairs and keypoints
    """

    def __init__(self, dataset_path: Path, return_type: str = None, indices: list[int] = None, rgb: bool = True):
        """

        @param imgs_path: Path to the original dataset
        @param transforms: transformations for the pre-processing, if desired
        @param return_type: The data can either be returned as a dict[Path] or a dict[np.ndarray]
        @param idxs: If desired, only specific indices can be used for the dataset creation (e.g. for train/val/test split)
        """

        super().__init__("FIRE", return_type, indices)

        self.ndim = 2
        self.spacing = (1, 1)
        self.image_shape = (2912, 2912)
        self.rgb = rgb

        self.has_keypoints = True
        self.has_segmentations = False

        self.images_path = dataset_path
        self.images_list = None
        self.keypoints_list = None
        self.__load_imgs_list__()
        self.__load_kps_list__()

        if indices is not None:  # create subsets for e.g. validation and training
            self.images_list = [self.images_list[i] for i in indices]
            self.keypoints_list = [self.keypoints_list[i] for i in indices]

    def preprocess(self, save_path: Path) -> None:
        """
        Preprocessing of the whole dataset
        @param save_path: The path where the preprocessing data should be saved. The same folder structure as in the
         original dataset will be created there automatically and after preprocessing, the data will be loaded from this path instead of the original one.
        @return: None
        """

        print("Not implemented")

    def __load_imgs_list__(self) -> None:
        """
        Reads the image files
        """

        path_images = self.images_path / "Images"
        all_paths = list(path_images.glob("*.*"))
        all_paths.sort()
        path_pairs = [[all_paths[i], all_paths[i + 1]]
                      for i in range(0, len(all_paths), 2)]
        self.images_list = path_pairs

    def __load_kps_list__(self):
        """
        Reads the keypoint files
        """
        self.keypoints_list = list()
        path_images = self.images_path / "Ground Truth"
        all_paths = list(path_images.glob("*.*"))
        all_paths.sort()
        self.keypoints_list = all_paths

    def _get_keypoint_pair_as_tensors(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        file_path = self.keypoints_list[idx]
        data = np.loadtxt(file_path.as_posix())
        coords_fixed = data[:, [0, 1]]
        coords_moving = data[:, [2, 3]]
        return torch.from_numpy(coords_fixed), torch.from_numpy(coords_moving)

    def _get_keypoint_pair_as_paths(self, idx: int) -> Tuple[Path, Path]:
        file_path = self.keypoints_list[idx]
        return file_path, file_path

    def _get_image_pair_as_tensors(self, idx: int):
        path_fixed = self.images_list[idx][0]
        path_moving = self.images_list[idx][1]
        fixed_imageixed = Image.open(path_fixed)
        # fixed_imageixed = Image.open(path_fixed).resize((256, 256))
        moving_imageoving = Image.open(path_moving)
        # moving_imageoving = Image.open(path_moving).resize((256, 256))
        if not self.rgb:
            fixed_imageixed = fixed_imageixed.convert('L')
            moving_imageoving = moving_imageoving.convert('L')
            fixed_imageixed = torch.from_numpy(np.array(fixed_imageixed))
            moving_imageoving = torch.tensor(np.array(moving_imageoving))
            fixed_imageixed = utils.normalize_tensor_to_0_1(fixed_imageixed)
            moving_imageoving = utils.normalize_tensor_to_0_1(
                moving_imageoving)

        if self.rgb:
            fixed_imageixed = torch.from_numpy(np.array(fixed_imageixed))
            moving_imageoving = torch.tensor(np.array(moving_imageoving))

        return fixed_imageixed, moving_imageoving

    def _get_image_pair_as_paths(self, idx):
        return self.images_path / self.images_list[idx][0], self.images_path / self.images_list[idx][1]

    def _get_image_pair(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns normalized greyscale or RGB images of shape (2912,2912, [3])
        :param idx:
        :return: (1,256,256) or (1, 3,256,256)
        """
        path_fixed = self.images_list[idx][0]
        path_moving = self.images_list[idx][1]
        fixed_imageixed = Image.open(path_fixed)
        # fixed_imageixed = Image.open(path_fixed).resize((256, 256))
        moving_imageoving = Image.open(path_moving)
        # moving_imageoving = Image.open(path_moving).resize((256, 256))
        if not self.rgb:
            fixed_imageixed = fixed_imageixed.convert('L')
            moving_imageoving = moving_imageoving.convert('L')
            fixed_imageixed = torch.tensor(fixed_imageixed)
            moving_imageoving = torch.tensor(moving_imageoving)
            fixed_imageixed = dataloader_utils.normalize_tensor_to_0_1(
                fixed_imageixed)
            moving_imageoving = dataloader_utils.normalize_tensor_to_0_1(
                moving_imageoving)

        if self.rgb:
            fixed_imageixed = torch.tensor(fixed_imageixed)
            moving_imageoving = torch.tensor(moving_imageoving)
            # fixed_imageixed = fixed_imageixed.movedim(2, 0)
            # moving_imageoving = moving_imageoving.movedim(2, 0)

        return fixed_imageixed, moving_imageoving


"""
Internal Datset structures for displacement fields and path pairs
"""


class BaselineTransformations(Dataset):
    """
    Dataloader for transformations generated by the baseline registration methods.
    """

    def __init__(self, path_transformations: Union[Path, List[Path]], idx: list[int] = None):
        """
            Initialize the Dataloader.

            When initilaizing with a path to results from baselines, it should be the path
            to the 'method' folder (in which 'deformations' and 'deformed' are stored).

            When initializing with a list of paths, each path should be the path to a transformation.

            The only criteria for transformation file names is that they have a '.' in them.
        """

        self.name = "BaselineTransformations"

        self.list_of_transformations = []

        if isinstance(path_transformations, Path):
            self.list_of_transformations = self.__load_transformations(
                path_transformations)
        elif isinstance(path_transformations, list) and all(isinstance(path, Path) for path in path_transformations):
            self.list_of_transformations = path_transformations
        else:
            raise TypeError(
                "path_transformations must be a Path or a list of Paths.")

        if len(self.list_of_transformations) == 0:
            raise ValueError("No transformations found.")

        # creae subset of transformations
        if idx is not None:
            self.list_of_transformations = [
                self.list_of_transformations[i] for i in idx]

    def __len__(self):
        """
            Return the number of transformations.
        """

        return len(self.list_of_transformations)

    def __getitem__(self, idx: int):
        """
            Return the path to the transformation at index idx.
        """

        return self.list_of_transformations[idx]

    def __load_transformations(self, path_result: Path) -> List[Path]:
        """
            Load the list of transformations.
        """

        list_of_transformations = None

        # get all deformations from path_result/deformations
        path_deformations = path_result / "deformations"

        list_of_transformations = list(path_deformations.glob("*.*"))

        # sort the list alphabetically
        list_of_transformations.sort()

        return list_of_transformations


class PathPairDataset():
    """
    A generic dataset for pairs of paths.
    """

    def __init__(self, path_pairs: List[tuple[Path, Path]]) -> None:
        self.path_pairs = path_pairs

        self.name = "PathPairDataset"

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx: int) -> tuple[Path, Path]:
        return self.path_pairs[idx]
