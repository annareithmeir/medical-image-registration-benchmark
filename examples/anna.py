import copy
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import numpy as np
import torch
from time import time
from tqdm import tqdm
import matplotlib
from pathlib import Path
import sys
import logging
import socket
import os
import time
import time
import warnings
import torch.nn.functional as F

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8

from registrationbaselines.core import utils  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.evaluation.evaluation import Evaluation  # nopep8
from registrationbaselines.training.train_voxelmorph_feature import VoxelmorphFeatureTraining  # nopep8
from registrationbaselines.registration.voxelmorph import VoxelmorphReg  # nopep8
from registrationbaselines.registration.bspline_feature import BSplineFeature  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8

from latent_space_registration.airlab.utils import image as iutils  # nopep8


def compute_grid(image_size, dtype=torch.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")


def warp_image(image, displacement):

    image_size = image.size

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)

    # warp image
    warped_image = F.grid_sample(image.image, displacement + grid)

    return warped_image


def deform_landmarks(moving_landmarks: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    # Map the moving landmarks to the fixed landmarks using the displacement field of shape (...,2) or (...,3)
    assert displacement.shape[-1] in [2, 3]
    if displacement.ndim == 4:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, :, 0], moving_landmarks.transpose())
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, :, 1], moving_landmarks.transpose())
        mov_lms_disp_z = map_coordinates(
            displacement[:, :, :, 2], moving_landmarks.transpose())
        mov_lms_disp = np.array(
            (mov_lms_disp_x, mov_lms_disp_y, mov_lms_disp_z)).transpose()
    if displacement.ndim == 3:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, 0], moving_landmarks.transpose())
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, 1], moving_landmarks.transpose())
        mov_lms_disp = np.array((mov_lms_disp_y, mov_lms_disp_x)).transpose()
    return moving_landmarks + mov_lms_disp


path_data = Path("/data/ACDC/database/")
train_dataset = data_loaders.ACDCDataset(path_data, return_mode="train_imgs2", normalize_mode=True, roi_only=True,
                                         dim_mode='2d-middle')
x, y = train_dataset.__getitem__(0)
x2 = copy.deepcopy(x)
y2 = copy.deepcopy(y)


val = 0

disp = torch.zeros((128, 128, 2))
disp[80:100, 80:100, :] = val

seg_y_pred = warp_image(iutils.Image(torch.tensor(y2).unsqueeze(0), (128, 128), (1, 1), (0, 0)),
                        torch.tensor(disp).unsqueeze(0))

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
plt.imshow(y2.squeeze())
axes.axis("on")
plt.legend()
fig.show()

plt.savefig(
    "/u/home/koeglf/Documents/code/registrationbaselines/examples/deformed_torch.png")
plt.close()

disp = np.zeros((128, 128, 2))
disp[80:100, 80:100, :] = val
# plt.imshow(x.squeeze())
# plt.show()
y = y.squeeze()
sitk_seg_x = sitk.GetImageFromArray(x.squeeze())
sitk_seg_y = sitk.GetImageFromArray(y.squeeze())
disp_sitk = sitk.GetImageFromArray(disp.astype(np.float64), isVector=True)
seg_y_pred = utils.deform_image(
    sitk_seg_y, sitk_seg_x, disp_sitk, sitk.sitkNearestNeighbor)
seg_y_pred = sitk.GetArrayFromImage(seg_y_pred)
fig, axes = plt.subplots(1, 1, figsize=(12, 6))
plt.imshow(seg_y_pred - sitk.GetArrayFromImage(sitk_seg_y))
axes.axis("on")
plt.legend()
fig.show()

plt.savefig(
    "/u/home/koeglf/Documents/code/registrationbaselines/examples/deformed_sitk.png")
plt.close()
