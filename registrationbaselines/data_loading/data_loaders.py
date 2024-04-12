import random
import torchio as tio
from torch.utils.data import Dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute().parent))

class DemoImageDataset(Dataset):
    def __init__(self, imgs_path: Path, transforms:list[str] = list(), target_transform=None):
        self.imgs_path = imgs_path
        self.img_shape = (192, 128, 192)
        self.transforms = transforms
        # self.target_transform = target_transform # pytorch transforms
        self.imgs_list = None

        self.__load_imgs_list__()

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx: int, return_as_subject=False):
        subject_dict = {
            "image_m": tio.ScalarImage(self.imgs_list[idx][0]),
            "image_f": tio.ScalarImage(self.imgs_list[idx][1]),
        }
        subject = tio.Subject(subject_dict)

        if "clip_bones" in self.transforms:
            clip = tio.Clamp(out_min=-400, out_max=1600)
            subject = clip(subject)

        if "normalize" in self.transforms:
            rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
            subject = rescale(subject)

        if "resample" in self.transforms:
            resample = tio.Resample(1)
            subject = resample(subject)

        img_m = subject["image_m"].data
        img_f = subject["image_f"].data

        # pytorch transform todo
        # if self.transform:
        #     image = self.transform(image)

        if return_as_subject:
            return subject
        else:
            return img_m, img_f

    def __load_imgs_list__(self):
        """
        in demo training set we assume that patient_0000.nii.gz is the moving image and patient_0001.nii.gz is the fixed image
        """

        self.imgs_list = list()
        for i in range(1, 4):
            file_m = 'LungCT_{:04d}_0000.nii.gz'.format(i)
            file_f = 'LungCT_{:04d}_0001.nii.gz'.format(i)
            self.imgs_list.append([self.imgs_path / file_m, self.imgs_path / file_f])

    def plot_random_image(self):
        rand_idx=random.randint(0,len(self)-1)
        # print(rand_idx)
        subject=self.__getitem__(rand_idx, return_as_subject=True)
        subject.plot()


