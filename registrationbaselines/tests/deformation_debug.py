from pathlib import Path
import sys
import logging
import socket
import os
import time
import numpy as np
import torch

import SimpleITK as sitk
import matplotlib.pyplot as plt

import registrationbaselines.io.io
import registrationbaselines.displacement.deform_objects


sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8
from registrationbaselines.registration.voxelmorph import VoxelMorph  # nopep8
from registrationbaselines.registration.syn_ants import SyNANTs  # nopep8
from registrationbaselines.registration.demons_sitk import DemonsSITK  # nopep8


path_displacement = Path(
    "/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/LungCT/DemonsSITK/DemonsSITK_708dc362-d897-4f5e-b65a-c0974e069d79/deformations/LungCT_0001_0001_deformation_to_LungCT_0001_0000.nii.gz")
path_displacement_sitk = Path(
    "/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/LungCT/DemonsSITK/DemonsSITK_708dc362-d897-4f5e-b65a-c0974e069d79/deformations/LungCT_0001_0001_deformation_to_LungCT_0001_0000_sitk.nii.gz")
path_original_warped = Path(
    "/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/LungCT/DemonsSITK/DemonsSITK_708dc362-d897-4f5e-b65a-c0974e069d79/deformed/LungCT_0001_0001_deformed_to_LungCT_0001_0000.nii.gz")
path_moving = Path(
    "/data/LungCT_preprocessed/imagesTr/LungCT_0001_0001.nii.gz")
path_fixed = Path(
    "/data/LungCT_preprocessed/imagesTr/LungCT_0001_0000.nii.gz")

sitk_displacement = sitk.ReadImage(
    path_displacement_sitk, sitk.sitkVectorFloat64)
sitk_original_warped = sitk.ReadImage(path_original_warped)
sitk_moving = sitk.ReadImage(path_moving)
sitk_fixed = sitk.ReadImage(path_fixed)

displacement_field_array = sitk.GetArrayFromImage(sitk_displacement)

# Convert the numpy array to a vector image
vector_displacement: sitk.Image = sitk.GetImageFromArray(
    -displacement_field_array, isVector=True)
outTx = sitk.DisplacementFieldTransform(vector_displacement)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(sitk_moving)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(outTx)

sitk_warped = resampler.Execute(sitk_moving)

ours_displacement = registrationbaselines.io.io.load_displacement(
    path_displacement)
ours_original_warped = registrationbaselines.io.io.load_image(
    path_original_warped)
ours_moving = registrationbaselines.io.io.load_image(path_moving)
ours_fixed = registrationbaselines.io.io.load_image(path_fixed)

ours_warped = registrationbaselines.displacement.deform_objects.deform_image(
    ours_moving, ours_displacement)

# cretae figure with three subplots in one row
fig, axs = plt.subplots(1, 5, figsize=(10, 5))
axs[0].imshow(ours_original_warped.permute(2, 1, 0)[50, ...], cmap='gray')
axs[0].set_title('Original Warped')
axs[1].imshow(ours_warped.permute(2, 1, 0)[50, ...], cmap='gray')
axs[1].set_title('Ours Warped')
axs[2].imshow(sitk.GetArrayFromImage(sitk_warped)[
              50, ...], cmap='gray')
axs[2].set_title('SITK Warped')
axs[3].imshow(ours_warped.permute(2, 1, 0)[50, ...] -
              ours_original_warped.permute(2, 1, 0)[50, ...], cmap='gray', vmin=-1, vmax=1)
axs[3].set_title('original - ours')
axs[4].imshow(sitk.GetArrayFromImage(sitk_warped)[50, ...] -
              ours_original_warped.permute(2, 1, 0)[50, ...].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
axs[4].set_title('original - SITK')
plt.show()
plt.savefig("deformation_debug.png")

diff_ori_vs_ours = np.sum(
    np.abs(ours_original_warped.cpu().numpy() - ours_warped.cpu().numpy()))
diff_ori_vs_sitk = np.sum(np.abs(sitk.GetArrayFromImage(
    sitk_original_warped) - sitk.GetArrayFromImage(sitk_warped)))
diff_sitk_vs_ours = np.sum(np.abs(sitk.GetArrayFromImage(
    sitk_warped) - ours_warped.permute(2, 1, 0).cpu().numpy()))

x = 0
