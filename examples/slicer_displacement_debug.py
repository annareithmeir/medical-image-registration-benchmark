from pathlib import Path
import sys
import os
import time
import random

from typing import Dict, Type, Union

import nibabel as nib
import SimpleITK as sitk
import torch
import numpy as np


sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.core import utils_niftyreg, utils_nifti  # nopep8
from registrationbaselines.io import load, save  # nopep8
from registrationbaselines.displacement import utils_displacement  # nopep8

path_disp_ori = Path(
    "/home/fryderyk/Downloads/displacement_debug_for_slicer/imagesTr/disp_original_copy.nii.gz")
path_disp_ori_intent = Path(
    "/home/fryderyk/Downloads/displacement_debug_for_slicer/imagesTr/disp_original_intent.nii.gz")
path_disp_baseline = Path(
    "/home/fryderyk/Downloads/displacement_debug_for_slicer/imagesTr/disp_baseline.nii.gz")

utils_niftyreg.convert_niftyreg_displacement_to_baseline_convention(path_disp_ori,
                                                                    path_disp_baseline)

# utils_nifti.set_intent_code(Path("/home/fryderyk/Downloads/displacement_debug_for_slicer/imagesTr/disp_original_intent.nii.gz"),
#                             'NIFTI_INTENT_DISPVECT')

disp_ori = sitk.ReadImage(str(path_disp_ori))
tensor_ori = torch.from_numpy(
    sitk.GetArrayFromImage(disp_ori)).to(torch.float64)

disp_baseline = sitk.ReadImage(str(path_disp_baseline))
tensor_baseline = torch.from_numpy(
    sitk.GetArrayFromImage(disp_baseline)).to(torch.float64)

tensor_baseline = tensor_baseline.squeeze(1)

tensor_baseline = tensor_baseline.permute(1, 2, 3, 0)

tensor_baseline = utils_displacement.displacement_to_unit_displacement(
    tensor_baseline)

shape = tuple(tensor_baseline.permute(2, 1, 0, 3).shape[:3])
scaling_tensor = torch.tensor(shape).unsqueeze(0).unsqueeze(0).unsqueeze(0)
tensor_baseline = tensor_baseline / 2 * scaling_tensor

torch.sum(torch.abs(tensor_ori-tensor_baseline))

a = torch.sum(torch.abs(tensor_ori-tensor_baseline))
print(a)
x = 0
# t_new = (t_old / tensor(shape)) * 2

# t_new / 2 = t_old / tensor(shape)

# (t_new / 2) * tensor(shape) = t_old
