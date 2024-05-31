import random
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
from torch.utils.data import Dataset
from tqdm import tqdm

"""
    Dataloader for the Learn2Reg LnugCT dataset.
    Since the test annotations are not available, we only load the training data with the corresponding segmentations and keypoints.
    The dataset has n=20 image pairs.
"""

# global clipping [min,max] values
# todo verify clipping parameters
WINDOW_BONES = [-400, 1600]
WINDOW_SOFT_TISSUE = [-150, 250]


class L2RLungCTDataset(Dataset):
    """
    Learn2Reg Lung CT dataset (available at https://learn2reg.grand-challenge.org/Datasets/)
    Intra-patient
    Currently only using the 20 training image pairs.
    """

    def __init__(self, dataset_path: Path, transforms: list[str] = None, return_type: str = "path",
                 indices: list[int] = None) -> None:
        """

        @param dataset_path: Path to the original or pre-processed dataset
        @param transforms: transformations for the pre-processing, if desired
        @param return_type: The data can either be returned as a dict[Path], dict[np.ndarray] or tuple of np.ndarray
        @param indices: If desired, only specific indices can be used for the dataset creation (e.g. for train/val/test split)
        """

        self.indices = indices
        self.images_path = dataset_path
        self.images_path_preprocessed = None
        self.transforms = transforms
        # self.target_transform = target_transform # todo: do we want torch.transforms too?
        self.ndim = 3
        self.spacing = (1.75, 1.25, 1.75)
        # after resampling to isotropic 1.75: (192, 138, 208)
        self.images_shape = (192, 192, 208)

        self.segmentation_labels = {
            0: "background",
            1: "lung"
        }

        assert return_type in ["path_dict", "np_array_dict", "np_arrays"]
        self.return_type = return_type

        self.images_list = None
        self.segmentations_list = None
        self.keypoints_list = None
        self.__load_images_list__()
        self.__load_segmentations_list__()
        self.__load_keypoints_list__()
        if indices is not None:  # create subsets for e.g. validation and training
            self.images_list = [self.images_list[i] for i in indices]
            self.segmentations_list = [self.segmentations_list[i] for i in indices]
            self.keypoints_list = [self.keypoints_list[i] for i in indices]

    def __len__(self) -> int:
        """

        @return: length of the dataset
        """
        return len(self.images_list)

    def __getitem__(self, idx: int) -> tuple | dict:
        """
        For accessing the individual image pairs
        @param idx: idx of image pair to return
        @return: Returns either a dict[Path] or a dict[np.ndarray] of imgs/segs/kps. In case of np arrays, the data is returned with shape (bs, h, w, d)
        """

        if self.return_type == "path_dict":
            item = {
                "images": [self.images_path / self.images_list[idx][0], self.images_path / self.images_list[idx][1]],
                "segmentations": [self.images_path / self.segmentations_list[idx][0],
                                  self.images_path / self.segmentations_list[idx][1]],
                "landmarks": [self.images_path / self.keypoints_list[idx][0],
                              self.images_path / self.keypoints_list[idx][1]]
            }
        elif self.return_type == "np_array_dict":  # np_array bsxhxwxd
            subject_dict = {
                "image_f": tio.ScalarImage(self.images_path / self.images_list[idx][0]),
                "image_m": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
                "seg_f": tio.ScalarImage(self.images_path / self.segmentations_list[idx][0]),
                "seg_m": tio.ScalarImage(self.images_path / self.segmentations_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            kp_f = np.genfromtxt(self.images_path / self.keypoints_list[idx][0],
                                 delimiter=',')

            kp_m = np.genfromtxt(self.images_path / self.keypoints_list[idx][1],
                                 delimiter=',')

            img_m = subject["image_m"].data.numpy()
            img_f = subject["image_f"].data.numpy()
            seg_m = subject["seg_m"].data.numpy()
            seg_f = subject["seg_f"].data.numpy()

            item = {
                "images": [img_f, img_m],
                "segmentations": [seg_f, seg_m],
                "landmarks": [kp_f, kp_m]
            }
        else:  # np_arrays of shape bsxhxwxd
            subject_dict = {
                "image_m": tio.ScalarImage(self.images_path / self.images_list[idx][0]),
                "image_f": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            img_m = subject["image_m"].data
            img_f = subject["image_f"].data

            item = (img_f, img_m)
        return item

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

        for idx in tqdm(range(len(self)), desc="Preprocessing (" + str(self.transforms) + ")", unit="iteration"):
            file_img_m = self.images_list[idx][0]
            file_img_f = self.images_list[idx][1]
            file_seg_m = self.segmentations_list[idx][0]
            file_seg_f = self.segmentations_list[idx][1]
            file_kp_m = self.keypoints_list[idx][0]
            file_kp_f = self.keypoints_list[idx][1]

            subject_dict = {
                "image_m": tio.ScalarImage(self.images_path / self.images_list[idx][0]),
                "image_f": tio.ScalarImage(self.images_path / self.images_list[idx][1]),
                "seg_m": tio.ScalarImage(self.images_path / self.segmentations_list[idx][0]),
                "seg_f": tio.ScalarImage(self.images_path / self.segmentations_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            kp_m = np.genfromtxt(
                self.images_path / self.keypoints_list[idx][0], delimiter=',')
            kp_f = np.genfromtxt(
                self.images_path / self.keypoints_list[idx][1], delimiter=',')

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
                self.images_shape = subject["image_m"].data.shape[1:]
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

        self.images_path = save_path
        print("From now on reading images from ", self.images_path)

    def __load_images_list__(self) -> None:
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

    def __load_segmentations_list__(self) -> None:
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

    def __load_keypoints_list__(self) -> None:
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

    def plot_random_image(self) -> None:
        """
        Plots a random image of the dataset including segmentations and keypoints
        """

        random_index = random.randint(0, len(self) - 1)
        tmp = self.return_type
        self.return_type = "path"
        item = self[random_index]
        self.return_type = tmp

        img_m = tio.ScalarImage(item["images"][0]).numpy().squeeze()
        img_f = tio.ScalarImage(item["images"][1]).numpy().squeeze()
        seg_m = tio.LabelMap(item["segmentations"][0]).numpy().squeeze()
        seg_f = tio.LabelMap(item["segmentations"][1]).numpy().squeeze()
        kp_m = np.genfromtxt(item["landmarks"][0], delimiter=',')
        kp_f = np.genfromtxt(item["landmarks"][1], delimiter=',')

        fig = plt.figure(figsize=(20, 12))
        image_size = self.images_shape
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
        plt.suptitle("idx: {}".format(random_index))
        plt.show()


class L2RAbdominalMRCTDataset(Dataset):
    """
    Learn2Reg Abdominal MR/CT dataset (available at https://learn2reg.grand-challenge.org/Datasets/)
    Intra-patient
    Currently only using the 8 paired and labeled images.
    8 in imagesTr and 8 in imagesTs. Only 8 in Ts are labeled.
    """

    def __init__(self, dataset_path: Path, transforms: list[str] = None, return_type: str = "path",
                 indices: list[int] = None) -> None:
        """
        
        @param dataset_path: path to the original or preprocessed dataset
        @param transforms: The preprocessing steps to be done (normalize, clip, resize)
        @param return_type: data type of return values. Either dict of paths, dict of np arrays or np arrays
        @param indices: list of indices which form the dataset (e.g. for train/val/test split)
        """

        self.indices = indices
        self.dataset_path = dataset_path
        self.dataset_path_preprocessed = None
        self.transforms = transforms
        # self.target_transform = self.config["torch_transforms"]
        self.ndim = 3
        self.spacing = (2, 2, 2)
        self.images_shape = (192, 160, 192)

        self.segmentation_labels = {
            0: "background",
            1: "liver",
            2: "spleen",
            3: "right kidney",
            4: "left kidney"
        }

        assert return_type in ["path_dict", "np_array_dict", "np_arrays"]
        self.return_type = return_type

        self.images_list = None
        self.segmentations_list = None
        self.__load_images_list__()
        self.__load_segmentations_list__()
        if indices is not None:  # create subsets for e.g. validation and training
            self.images_list = [self.images_list[i] for i in indices]
            self.segmentations_list = [self.segmentations_list[i] for i in indices]
            # print("sliced:", idxs)

    def __len__(self) -> int:
        """
        Returns length of the dataset
        @return:
        """
        return len(self.images_list)

    def __getitem__(self, idx: int) -> tuple | dict:
        """
        For accessing the individual image pairs
        @param idx: idx of image pair to return
        @return: Returns either a dict[Path] or a dict[np.ndarray] of imgs/segs/kps. In case of np arrays, the data is returned with shape (bs, h, w, d)
        """

        if self.return_type == "path_dict":
            item = {
                "images": [self.dataset_path / self.images_list[idx][0], self.dataset_path / self.images_list[idx][1]],
                "segmentations": [self.dataset_path / self.segmentations_list[idx][0],
                                  self.dataset_path / self.segmentations_list[idx][1]]
            }
        elif self.return_type == "np_array_dict":  # np_array bsxhxwxd
            subject_dict = {
                "image_f": tio.ScalarImage(self.dataset_path / self.images_list[idx][0]),
                "image_m": tio.ScalarImage(self.dataset_path / self.images_list[idx][1]),
                "seg_f": tio.ScalarImage(self.dataset_path / self.segmentations_list[idx][0]),
                "seg_m": tio.ScalarImage(self.dataset_path / self.segmentations_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            img_m = subject["image_m"].data
            img_f = subject["image_f"].data
            seg_m = subject["seg_m"].data
            seg_f = subject["seg_f"].data

            item = {
                "images": [img_f, img_m],
                "segmentations": [seg_f, seg_m]
            }
        else:  # np_arrays of shape bsxhxwxd
            subject_dict = {
                "image_f": tio.ScalarImage(self.dataset_path / self.images_list[idx][0]),
                "image_m": tio.ScalarImage(self.dataset_path / self.images_list[idx][1])
            }
            subject = tio.Subject(subject_dict)

            img_m = subject["image_m"].data.numpy()
            img_f = subject["image_f"].data.numpy()

            item = (img_f, img_m)
        return item

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

        for idx in tqdm(range(len(self)), desc="Preprocessing (" + str(self.transforms) + ")", unit="iteration"):
            file_img_m = self.images_list[idx][1]
            file_img_f = self.images_list[idx][0]
            file_seg_m = self.segmentations_list[idx][1]
            file_seg_f = self.segmentations_list[idx][0]

            subject_dict = {
                "image_f": tio.ScalarImage(self.dataset_path / self.images_list[idx][0]),
                "image_m": tio.ScalarImage(self.dataset_path / self.images_list[idx][1]),
                "seg_f": tio.ScalarImage(self.dataset_path / self.segmentations_list[idx][0]),
                "seg_m": tio.ScalarImage(self.dataset_path / self.segmentations_list[idx][1])
            }
            subject = tio.Subject(subject_dict)

            if "clip_bones" in self.transforms:
                clip = tio.Clamp(
                    out_min=WINDOW_BONES[0], out_max=WINDOW_BONES[1])
                subject["image_ct"] = clip(subject["image_ct"])

            if "clip_soft_tissue" in self.transforms:
                clip = tio.Clamp(
                    out_min=WINDOW_SOFT_TISSUE[0], out_max=WINDOW_SOFT_TISSUE[1])
                subject["image_ct"] = clip(subject["image_ct"])

            if "normalize" in self.transforms:
                rescale = tio.RescaleIntensity(
                    out_min_max=(0, 1), percentiles=(0, 100))
                subject = rescale(subject)

            if "resample" in self.transforms:
                resample = tio.Resample(1)
                subject = resample(subject)
                self.images_shape = subject["image_mr"].data.shape[1:]
                self.spacing = (1, 1, 1)

            # save preprocessed images
            subject["image_m"].save(save_path / file_img_m)
            subject["image_f"].save(save_path / file_img_f)
            subject["seg_m"].save(save_path / file_seg_m)
            subject["seg_f"].save(save_path / file_seg_f)

        self.dataset_path = save_path
        print("From now on reading images from ", self.dataset_path)

    def __load_images_list__(self) -> None:
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

    def __load_segmentations_list__(self) -> None:
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

    def plot_random_image(self) -> None:
        """
        Plots a random image of the dataset including segmentations
        """

        random_index = random.randint(0, len(self) - 1)
        tmp = self.return_type
        self.return_type = "path_dict"
        item = self[random_index]
        self.return_type = tmp

        img_f = tio.ScalarImage(item["images"][0]).numpy().squeeze()
        img_m = tio.ScalarImage(item["images"][1]).numpy().squeeze()
        seg_f = tio.LabelMap(item["segmentations"][0]).numpy().squeeze()
        seg_m = tio.LabelMap(item["segmentations"][1]).numpy().squeeze()

        fig = plt.figure(figsize=(20, 12))
        image_size = self.images_shape
        slices = [int(image_size[0] / 2), int(image_size[1] / 2),
                  int(image_size[2] / 2)]

        for a in range(0, 3):
            # moving image
            ax = fig.add_subplot(2, 3, a + 1)
            if a == 0:
                slice_image_m = img_m[slices[a], :, :]
                slice_segmentation_m = seg_m[slices[a], :, :]
            if a == 1:
                slice_image_m = img_m[:, slices[a], :]
                slice_segmentation_m = seg_m[:, slices[a], :]
            if a == 2:
                slice_image_m = img_m[:, :, slices[a]]
                slice_segmentation_m = seg_m[:, :, slices[a]]

            plt.imshow(slice_image_m, cmap='gray')
            plt.colorbar()
            plt.imshow(slice_segmentation_m, alpha=0.3)
            plt.title("moving")

            # fixed image
            ax = fig.add_subplot(2, 3, a + 4)
            if a == 0:
                slice_image_f = img_f[slices[a], :, :]
                slice_segmentation_f = seg_f[slices[a], :, :]
            if a == 1:
                slice_image_f = img_f[:, slices[a], :]
                slice_segmentation_f = seg_f[:, slices[a], :]
            if a == 2:
                slice_image_f = img_f[:, :, slices[a]]
                slice_segmentation_f = seg_f[:, :, slices[a]]

            plt.imshow(slice_image_f, cmap='gray')
            plt.title("fixed")
            plt.colorbar()
            plt.imshow(slice_segmentation_f, alpha=0.3)
            # plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.suptitle("idx: {}".format(random_index))
        plt.show()


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
