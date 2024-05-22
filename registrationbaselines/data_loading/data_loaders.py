import random
import torchio as tio
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# class DemoImageDataset(Dataset):
#     def __init__(self, imgs_path: Path, transforms:list[str] = list(), target_transform=None):
#         self.imgs_path = imgs_path
#         self.ndim = 3
#         self.transforms = transforms
#         # self.target_transform = target_transform # pytorch transforms
#
#         if "resample" in transforms:
#             self. spacing = (1,1,1)
#             self.img_shape = (336, 160, 336)
#         else:
#             self.spacing = (1,1,1)
#             self.img_shape = (192, 128, 192)
#         self.imgs_list = None
#
#         self.__load_imgs_list__()
#
#     def __len__(self):
#         return len(self.imgs_list)

    # def __getitem__(self, idx: int, return_as_subject=False):
    #     subject_dict = {
    #         "image_m": tio.ScalarImage(self.imgs_list[idx][0]),
    #         "image_f": tio.ScalarImage(self.imgs_list[idx][1]),
    #     }
    #     subject = tio.Subject(subject_dict)
    #
    #     if "clip_bones" in self.transforms:
    #         clip = tio.Clamp(out_min=-400, out_max=1600)
    #         subject = clip(subject)
    #
    #     if "normalize" in self.transforms:
    #         rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
    #         subject = rescale(subject)
    #
    #     if "resample" in self.transforms:
    #         resample = tio.Resample(1)
    #         subject = resample(subject)
    #
    #
    #     img_m = subject["image_m"].data
    #     img_f = subject["image_f"].data
    #
    #     # pytorch transform todo
    #     # if self.transform:
    #     #     image = self.transform(image)
    #
    #     if return_as_subject:
    #         return subject
    #     else:
    #         return img_m, img_f
    #
    # def __load_imgs_list__(self):
    #     """
    #     in demo training set we assume that patient_0000.nii.gz is the moving image and patient_0001.nii.gz is the fixed image
    #     """
    #
    #     self.imgs_list = list()
    #     for i in range(1, 4):
    #         file_m = 'LungCT_{:04d}_0000.nii.gz'.format(i)
    #         file_f = 'LungCT_{:04d}_0001.nii.gz'.format(i)
    #         self.imgs_list.append([self.imgs_path / file_m, self.imgs_path / file_f])
    #
    # def plot_random_image(self):
    #     rand_idx=random.randint(0,len(self)-1)
    #     subject=self.__getitem__(rand_idx, return_as_subject=True)
    #     subject.plot()


class L2RLungCTDataset(Dataset):

    def __init__(self, imgs_path: Path, transforms: list[str] = list(), return_type: str = "path", idxs: list[int] = None):

        self.idxs = idxs
        self.imgs_path = imgs_path
        self.imgs_path_preprocessed = None
        self.transforms = transforms
        # self.target_transform = target_transform # todo: do we want torch.transforms too?
        self.ndim = 3
        if "resample" in transforms:
            # resample to uniform pixel size of 2mm
            self. spacing = (1.75, 1.75, 1.75)
            self.img_shape = (192, 192, 208)
        else:
            self. spacing = (1.75, 1.25, 1.75)
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
        if idxs is not None: # create subsets for e.g. validation and training
            self.imgs_list=[self.imgs_list[i] for i in idxs]
            self.segs_list=[self.segs_list[i] for i in idxs]
            self.kps_list=[self.kps_list[i] for i in idxs]
            # print("sliced:", idxs)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx: int):

        if self.return_type == "path":
            dict={
                "imgs": [self.imgs_list[idx][0],self.imgs_list[idx][1]],
                "segs": [self.segs_list[idx][0],self.segs_list[idx][1]],
                "kps": [self.kps_list[idx][0], self.kps_list[idx][1]]
            }
        else: # np_array bsxhxwxd
            subject_dict = {
                "image_m": tio.ScalarImage(self.imgs_list[idx][0]),
                "image_f": tio.ScalarImage(self.imgs_list[idx][1]),
                "seg_m": tio.ScalarImage(self.segs_list[idx][0]),
                "seg_f": tio.ScalarImage(self.segs_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            kp_m = self.kps_list[idx][0]
            kp_f = self.kps_list[idx][1]

            img_m = subject["image_m"].data
            img_f = subject["image_f"].data
            seg_m = subject["seg_m"].data
            seg_f = subject["seg_f"].data

            dict = {
                "imgs": [img_m,img_f],
                "segs": [seg_m, seg_f],
                "kps": [kp_m, kp_f]
            }
        return dict

    def preprocess(self, save_path: Path) -> None:

        self.imgs_path = save_path

        for idx in range(len(self)):
            file_img_m = "/".join(self.imgs_list[idx][0].split("/")[-2:])
            file_img_f = "/".join(self.imgs_list[idx][1].split("/")[-2:])
            file_seg_m = "/".join(self.segs_list[idx][0].split("/")[-2:])
            file_seg_f = "/".join(self.segs_list[idx][1].split("/")[-2:])
            file_kp_m = "/".join(self.kps_list[idx][0].split("/")[-2:])
            file_kp_f = "/".join(self.kps_list[idx][1].split("/")[-2:])

            subject_dict = {
                "image_m": tio.ScalarImage(self.imgs_list[idx][0]),
                "image_f": tio.ScalarImage(self.imgs_list[idx][1]),
                "seg_m": tio.ScalarImage(self.segs_list[idx][0]),
                "seg_f": tio.ScalarImage(self.segs_list[idx][1]),
            }
            subject = tio.Subject(subject_dict)

            kp_m = np.genfromtxt(self.kps_list[idx][0], delimiter=',')
            kp_f = np.genfromtxt(self.kps_list[idx][0], delimiter=',')

            if "clip_bones" in self.transforms:
                print("clipping bones ...")
                clip = tio.Clamp(out_min=-400, out_max=1600)
                subject = clip(subject)

            if "normalize" in self.transforms:
                print("normalizing ...")
                rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
                subject = rescale(subject)

            if "resample" in self.transforms:
                print("resampling ...")
                resample = tio.Resample(1.75)
                subject = resample(subject)
                self.img_shape = subject["image_m"].data.shape[1:]

                # after resmpling, the keypoints coordinates need to be adapted
                #kp_m = kp_m[:, [2, 1, 0]]
                #kp_f = kp_f[:, [2, 1, 0]]
                kp_m[:, 1] = kp_m[:, 1] * 1.25 / 1.75
                kp_f[:, 1] = kp_f[:, 1] * 1.25 / 1.75

            # save preprocessed images
            subject["image_m"].save(save_path / file_img_m)
            subject["image_f"].save(save_path / file_img_f)
            subject["seg_m"].save(save_path / file_seg_m)
            subject["seg_f"].save(save_path / file_seg_f)
            np.savetxt(save_path / file_kp_m, kp_m, delimiter=",")
            np.savetxt(save_path / file_kp_f, kp_f, delimiter=",")

    def __load_imgs_list__(self):
        self.imgs_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = "imagesTr/" + file_str + "_0001.nii.gz"
            file_f = "imagesTr/" + file_str + "_0000.nii.gz"
            self.imgs_list.append([self.imgs_path / file_m, self.imgs_path / file_f])

    def __load_segs_list__(self):
        self.segs_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = "masksTr/" + file_str + "_0001.nii.gz"
            file_f = "masksTr/" + file_str + "_0000.nii.gz"
            self.segs_list.append([self.imgs_path / file_m, self.imgs_path / file_f])

    def __load_kps_list__(self):
        self.kps_list = list()
        for i in range(1, 21):
            file_str = "LungCT_" + str(i).zfill(4)
            file_m = "keypointsTr/" + file_str + "_0001.csv"
            file_f = "keypointsTr/" + file_str + "_0000.csv"
            self.kps_list.append([self.imgs_path / file_m, self.imgs_path / file_f])

    def plot_random_image(self) -> None:

        rand_idx = random.randint(0, len(self) - 1)
        tmp = self.return_type
        self.return_type="path"
        item = self[rand_idx]
        self.return_type=tmp

        img_m = tio.ScalarImage(item["imgs"][0]).numpy().squeeze()
        img_f = tio.ScalarImage(item["imgs"][1]).numpy().squeeze()
        seg_m = tio.LabelMap(item["segs"][0]).numpy().squeeze()
        seg_f = tio.LabelMap(item["segs"][1]).numpy().squeeze()
        kp_m = np.genfromtxt(item["kps"][0], delimiter=',')
        kp_f = np.genfromtxt(item["kps"][1], delimiter=',')

        fig = plt.figure(figsize=(20, 12))
        image_size = self.img_shape
        slices = [int(image_size[0] / 2), int(image_size[1] / 2), int(image_size[2] / 2)]

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
                slice_kp_m = slice_kp_m[:, [0,2]]
            if a == 2:
                slice_img_m = img_m[:, :, slices[a]]
                slice_seg_m = seg_m[:, :, slices[a]]
                slice_kp_m = kp_m[np.where(abs(kp_m[:, 2] - slices[a]) <= 0.5)]
                slice_kp_m = slice_kp_m[:, :-1]

            plt.imshow(slice_img_m, cmap='gray')
            plt.colorbar()
            plt.imshow(slice_seg_m, alpha=0.3)
            plt.scatter(slice_kp_m[:, 1], slice_kp_m[:, 0], marker='x', c='red')
            #plt.gca().invert_yaxis()

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
                slice_kp_f = slice_kp_f[:, [0,2]]
            if a == 2:
                slice_img_f = img_f[:, :, slices[a]]
                slice_seg_f = seg_f[:, :, slices[a]]
                slice_kp_f = kp_f[np.where(abs(kp_f[:, 2] - slices[a]) <= 0.5)]
                slice_kp_f = slice_kp_f[:, :-1]

            plt.imshow(slice_img_f, cmap='gray')
            plt.colorbar()
            plt.imshow(slice_seg_f, alpha=0.3)
            plt.scatter(slice_kp_f[:, 1], slice_kp_f[:, 0], marker='x', c='red')
            #plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.suptitle("idx: {}".format(rand_idx))
        plt.show()



