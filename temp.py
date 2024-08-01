from pathlib import Path

import SimpleITK as sitk
import numpy as np

path_moving = Path(
    r"/home/fryderyk/Documents/data/LungCT_preprocessed/imagesTr/LungCT_0001_0001.nii.gz")
path_warped = Path(
    r"/home/fryderyk/Documents/code/registrationbaselines/tmp/deformation_debugging/LungCT/BSplineNiftyReg/deformed/LungCT_0001_0001_deformed_to_LungCT_0001_0000.nii.gz")
path_deformation = Path(
    r"/home/fryderyk/Documents/code/registrationbaselines/tmp/deformation_debugging/LungCT/BSplineNiftyReg/deformations/LungCT_0001_0001_deformation_to_LungCT_0001_0000.nii.gz")

image_moving = sitk.ReadImage(str(path_moving))
image_warped = sitk.ReadImage(str(path_warped))
image_deformation = sitk.ReadImage(str(path_deformation))

array_moving = sitk.GetArrayFromImage(image_moving)
array_warped = sitk.GetArrayFromImage(image_warped)
array_deformation = sitk.GetArrayFromImage(image_deformation)

print("sitk vs numpy")
print(f"moving: {image_moving.GetSize()} <-> {array_moving.shape}")
print(f"warped: {image_warped.GetSize()} <-> {array_warped.shape}")
print(
    f"deformation: {image_deformation.GetSize()} <-> {array_deformation.shape}")

"""

moving: (192, 138, 208) <-> (208, 138, 192)
warped: (192, 138, 208) <-> (208, 138, 192)
deformation: (192, 138, 208, 1, 3) <-> (3, 1, 208, 138, 192)

"""

x = 0
