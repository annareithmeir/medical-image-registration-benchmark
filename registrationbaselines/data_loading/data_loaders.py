import random
from pathlib import Path
import json

from typing import List, Union, Optional, Iterator, Tuple

import torchio as tio
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
    Dataloader for the Learn2Reg LnugCT dataset.
    Since the test annotations are not available, we only load the training data with the corresponding segmentations and keypoints.
    The dataset has n=20 image pairs.
"""


class L2RLungCTDataset(Dataset):

    def __init__(self, imgs_path: Path, transforms: list[str] = list(), return_type: str = "path", idxs: list[int] = None):
        """

        @param imgs_path: Path to the original dataset
        @param transforms: transformations for the pre-processing, if desired
        @param return_type: The data can either be returned as a dict[Path] or a dict[np.ndarray]
        @param idxs: If desired, only specific indices can be used for the dataset creation (e.g. for train/val/test split)
        """

        self.idxs = idxs
        self.imgs_path = imgs_path
        self.imgs_path_preprocessed = None
        self.transforms = transforms
        # self.target_transform = target_transform # todo: do we want torch.transforms too?
        self.ndim = 3
        self. spacing = (1.75, 1.25, 1.75)
        # after resampling to isotropic 1.75: (192, 138, 208)
        self.img_shape = (192, 192, 208)

        self.seg_labels = {
            0: "background",
            1: "lung"
        }

        assert return_type in ["path", "np_array"]
        self.return_type = return_type

        self.imgs_list = None
        self.segs_list = None
        self.kps_list = None
        self.__load_imgs_list__()
        self.__load_segs_list__()
        self.__load_kps_list__()
        if idxs is not None:  # create subsets for e.g. validation and training
            self.imgs_list = [self.imgs_list[i] for i in idxs]
            self.segs_list = [self.segs_list[i] for i in idxs]
            self.kps_list = [self.kps_list[i] for i in idxs]
            # print("sliced:", idxs)

    def __len__(self) -> int:
        """

        @return: length of the dataset
        """
        return len(self.imgs_list)

    def __getitem__(self, idx: int):
        """
        For accessing the individual image pairs
        @param idx: idx of image pair to return
        @return: Returns either a dict[Path] or a dict[np.ndarray] of imgs/segs/kps. In case of np arrays, the data is returned with shape (bs, h, w, d)
        """

        if self.return_type == "path":
            item_dict = {
                "imgs": [self.imgs_path / self.imgs_list[idx][0], self.imgs_path / self.imgs_list[idx][1]],
                "segs": [self.imgs_path / self.segs_list[idx][0], self.imgs_path / self.segs_list[idx][1]],
                "kps": [self.imgs_path / self.kps_list[idx][0], self.imgs_path / self.kps_list[idx][1]]
            }
        else:  # np_array bsxhxwxd
            subject_dict = {
                "image_m": tio.ScalarImage(self.imgs_path / self.imgs_list[idx][0]),
                "image_f": tio.ScalarImage(self.imgs_path / self.imgs_list[idx][1]),
                "seg_m": tio.ScalarImage(self.imgs_path / self.segs_list[idx][0]),
                "seg_f": tio.ScalarImage(self.imgs_path / self.segs_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            kp_m = self.imgs_path / self.kps_list[idx][0]
            kp_f = self.imgs_path / self.kps_list[idx][1]

            img_m = subject["image_m"].data
            img_f = subject["image_f"].data
            seg_m = subject["seg_m"].data
            seg_f = subject["seg_f"].data

            item_dict = {
                "imgs": [img_m, img_f],
                "segs": [seg_m, seg_f],
                "kps": [kp_m, kp_f]
            }
        return item_dict

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

        for idx in tqdm(range(len(self)), desc="Preprocessing ("+str(self.transforms)+")", unit="iteration"):
            file_img_m = self.imgs_list[idx][0]
            file_img_f = self.imgs_list[idx][1]
            file_seg_m = self.segs_list[idx][0]
            file_seg_f = self.segs_list[idx][1]
            file_kp_m = self.kps_list[idx][0]
            file_kp_f = self.kps_list[idx][1]

            subject_dict = {
                "image_m": tio.ScalarImage(self.imgs_path / self.imgs_list[idx][0]),
                "image_f": tio.ScalarImage(self.imgs_path / self.imgs_list[idx][1]),
                "seg_m": tio.ScalarImage(self.imgs_path / self.segs_list[idx][0]),
                "seg_f": tio.ScalarImage(self.imgs_path / self.segs_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            kp_m = np.genfromtxt(
                self.imgs_path / self.kps_list[idx][0], delimiter=',')
            kp_f = np.genfromtxt(
                self.imgs_path / self.kps_list[idx][1], delimiter=',')

            if "clip_bones" in self.transforms:
                clip = tio.Clamp(out_min=-400, out_max=1600)
                subject = clip(subject)

            if "normalize" in self.transforms:
                rescale = tio.RescaleIntensity(
                    out_min_max=(0, 1), percentiles=(0, 100))
                subject = rescale(subject)

            if "resample" in self.transforms:
                resample = tio.Resample(1.75)
                subject = resample(subject)
                self.img_shape = subject["image_m"].data.shape[1:]
                self.spacing = (1.75, 1.75, 1.75)

                # after resmpling, the keypoints coordinates need to be adapted
                kp_m[:, 1] = kp_m[:, 1] * 1.25 / 1.75
                kp_f[:, 1] = kp_f[:, 1] * 1.25 / 1.75

            # save preprocessed images
            subject["image_m"].save(save_path / file_img_m)
            subject["image_f"].save(save_path / file_img_f)
            subject["seg_m"].save(save_path / file_seg_m)
            subject["seg_f"].save(save_path / file_seg_f)
            np.savetxt(save_path / file_kp_m, kp_m, delimiter=",")
            np.savetxt(save_path / file_kp_f, kp_f, delimiter=",")

        self.imgs_path = save_path
        print("From now on reading images from ", self.imgs_path)

    def __load_imgs_list__(self) -> None:
        """
        Reads the image files
        """
        self.imgs_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = Path("imagesTr/" + file_str + "_0001.nii.gz")
            file_f = Path("imagesTr/" + file_str + "_0000.nii.gz")
            self.imgs_list.append([file_m, file_f])

    def __load_segs_list__(self):
        """
        Reads the segmentation files
        """
        self.segs_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = Path("masksTr/" + file_str + "_0001.nii.gz")
            file_f = Path("masksTr/" + file_str + "_0000.nii.gz")
            self.segs_list.append([file_m, file_f])

    def __load_kps_list__(self):
        """
        Reads the keypoint files
        """
        self.kps_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = Path("keypointsTr/" + file_str + "_0001.csv")
            file_f = Path("keypointsTr/" + file_str + "_0000.csv")
            self.kps_list.append([file_m, file_f])

    def plot_random_image(self) -> None:
        """
        Plots a random image of the dataset including segmentations and keypoints
        """

        rand_idx = random.randint(0, len(self) - 1)
        tmp = self.return_type
        self.return_type = "path"
        item = self[rand_idx]
        self.return_type = tmp

        img_m = tio.ScalarImage(item["imgs"][0]).numpy().squeeze()
        img_f = tio.ScalarImage(item["imgs"][1]).numpy().squeeze()
        seg_m = tio.LabelMap(item["segs"][0]).numpy().squeeze()
        seg_f = tio.LabelMap(item["segs"][1]).numpy().squeeze()
        kp_m = np.genfromtxt(item["kps"][0], delimiter=',')
        kp_f = np.genfromtxt(item["kps"][1], delimiter=',')

        fig = plt.figure(figsize=(20, 12))
        image_size = self.img_shape
        slices = [int(image_size[0] / 2), int(image_size[1] / 2),
                  int(image_size[2] / 2)]

        for a in range(0, 3):
            # moving image
            ax = fig.add_subplot(2, 3, a + 1)
            if a == 0:
                slice_img_m = img_m[slices[a], :, :]
                slice_seg_m = seg_m[slices[a], :, :]
                slice_kp_m = kp_m[np.where(abs(kp_m[:, 0] - slices[a]) <= 0.5)]
                slice_kp_m = slice_kp_m[:, 1:]
            if a == 1:
                slice_img_m = img_m[:, slices[a], :]
                slice_seg_m = seg_m[:, slices[a], :]
                slice_kp_m = kp_m[np.where(abs(kp_m[:, 1] - slices[a]) <= 0.5)]
                slice_kp_m = slice_kp_m[:, [0, 2]]
            if a == 2:
                slice_img_m = img_m[:, :, slices[a]]
                slice_seg_m = seg_m[:, :, slices[a]]
                slice_kp_m = kp_m[np.where(abs(kp_m[:, 2] - slices[a]) <= 0.5)]
                slice_kp_m = slice_kp_m[:, :-1]

            plt.imshow(slice_img_m, cmap='gray')
            plt.colorbar()
            plt.imshow(slice_seg_m, alpha=0.3)
            plt.scatter(slice_kp_m[:, 1],
                        slice_kp_m[:, 0], marker='x', c='red')
            # plt.gca().invert_yaxis()

            # fixed image
            ax = fig.add_subplot(2, 3, a + 4)
            if a == 0:
                slice_img_f = img_f[slices[a], :, :]
                slice_seg_f = seg_f[slices[a], :, :]
                slice_kp_f = kp_f[np.where(abs(kp_f[:, 0] - slices[a]) <= 0.5)]
                slice_kp_f = slice_kp_f[:, 1:]
            if a == 1:
                slice_img_f = img_f[:, slices[a], :]
                slice_seg_f = seg_f[:, slices[a], :]
                slice_kp_f = kp_f[np.where(abs(kp_f[:, 1] - slices[a]) <= 0.5)]
                slice_kp_f = slice_kp_f[:, [0, 2]]
            if a == 2:
                slice_img_f = img_f[:, :, slices[a]]
                slice_seg_f = seg_f[:, :, slices[a]]
                slice_kp_f = kp_f[np.where(abs(kp_f[:, 2] - slices[a]) <= 0.5)]
                slice_kp_f = slice_kp_f[:, :-1]

            plt.imshow(slice_img_f, cmap='gray')
            plt.colorbar()
            plt.imshow(slice_seg_f, alpha=0.3)
            plt.scatter(slice_kp_f[:, 1],
                        slice_kp_f[:, 0], marker='x', c='red')
            # plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.suptitle("idx: {}".format(rand_idx))
        plt.show()


class BaselineTransformations():
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

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx: int) -> tuple[Path, Path]:
        return self.path_pairs[idx]


# class L2RLungCTDataset():
#     def __init__(self, path_root: Path):
#         """
#         Initialize the dataset.
#
#         params:
#         path_root (Union[Path, List[List[Path]]]): Path to the root directory containing the images
#         or a list of lists of paths to the images.
#         """
#
#         self.path_list = self._load_imgs_list(path_root)
#
#     def __len__(self):
#         return len(self.path_list)
#
#     def __getitem__(self,
#                     idx,
#                     return_segmentation: Optional[bool] = None) -> tuple[Path, Path]:
#
#         fixed_path = None
#         moving_path = None
#
#         if return_segmentation:
#             fixed_path = self.path_list[idx]["mask_f"]
#             moving_path = self.path_list[idx]["mask_m"]
#         else:
#             fixed_path = self.path_list[idx]["image_f"]
#             moving_path = self.path_list[idx]["image_m"]
#
#         return fixed_path, moving_path
#
#     def __iter__(self):
#         raise NotImplementedError(
#             "This method is not implemented. Use the method __getitem__ instead.")
#
#     def _load_imgs_list(self, path_root: Path):
#
#         path_list = []
#
#         for i in range(1, 21):
#             file_str = "LungCT_" + str(i).zfill(4)
#
#             file_image_f = "imagesTr/" + file_str + "_0000.nii.gz"
#             file_image_m = "imagesTr/" + file_str + "_0001.nii.gz"
#
#             flie_mask_f = "masksTr/" + file_str + "_0000.nii.gz"
#             file_mask_m = "masksTr/" + file_str + "_0001.nii.gz"
#
#             path_list.append({
#                 "image_f": path_root / file_image_f,
#                 "image_m": path_root / file_image_m,
#                 "mask_f": path_root / flie_mask_f,
#                 "mask_m": path_root / file_mask_m
#             })
#
#         return path_list
