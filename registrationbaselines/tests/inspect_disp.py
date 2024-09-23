from pathlib import Path
import sys

import SimpleITK as sitk
import nibabel as nib
import torch
import numpy as np

import registrationbaselines.displacement.deform_objects


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.core import utils, utils_nifti  # nopep8


###########################################################################################################################################
# REGISTER WITH NiftyReg
###########################################################################################################################################

path_fixed = Path(
    "registrationbaselines/tests/images/fixed_x_11.nii.gz")
path_moving = Path(
    "registrationbaselines/tests/images/moving_x_11.nii.gz")
shape = nib.load(path_fixed).shape

result_registration = registrationbaselines.displacement.deform_objects.register_niftyreg(
    path_fixed, path_moving)

# result_registration = {"warped": Path("registrationbaselines/tests/images/moving_x_11_warped.nii.gz"),
#                        "deformation": Path("registrationbaselines/tests/images/deformation.nii.gz")}

NUMBER_OF_PIXELS = np.prod(shape)
TRUE_WARPED = nib.load(result_registration["warped"]).get_fdata()
TRUE_WARPED_SITK = sitk.GetArrayFromImage(
    sitk.ReadImage(result_registration["warped"]))


# get amount of valeues in TRUE_WARPED that are not nan
TRUE_WARPED_NOT_NAN = np.sum(~np.isnan(TRUE_WARPED))

###########################################################################################################################################
# LOAD WITH NiftyReg AND DEFORM WITH NiftyReg
###########################################################################################################################################

result_warping_manual_path = registrationbaselines.displacement.deform_objects.deform_image_niftyreg_path(path_moving,
                                                                                                     result_registration["deformation"])
diff = TRUE_WARPED - nib.load(result_warping_manual_path).get_fdata()

print("\nLOAD WITH NiftyReg AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# rename result_warping_manual_path to "warped_niftyreg.nii.gz"
Path(result_warping_manual_path).rename(
    "registrationbaselines/tests/images/warped_niftyreg.nii.gz")
result_warping_manual_path = Path(
    "registrationbaselines/tests/images/warped_niftyreg.nii.gz")

utils.print_histogram(torch.from_numpy(
    nib.load(result_warping_manual_path).get_fdata()), 10)

###########################################################################################################################################
# LOAD WITH nibabel AND DEFORM WITH NiftyReg
###########################################################################################################################################

nib_deformation = nib.load(result_registration["deformation"])
nib_moving = nib.load(path_moving)

result_warping_manual_nibabel = registrationbaselines.displacement.deform_objects.deform_image_niftyreg_nibabel(nib_moving,
                                                                                                           nib_deformation)

diff = TRUE_WARPED - result_warping_manual_nibabel.get_fdata()
print("\nLOAD WITH nibabel AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# save warped
nib.save(result_warping_manual_nibabel,
         "registrationbaselines/tests/images/warped_nib.nii.gz")

utils.print_histogram(torch.from_numpy(
    result_warping_manual_nibabel.get_fdata()), 10)

###########################################################################################################################################
# LOAD WITH NUMPY AND DEFORM WITH NiftyReg
###########################################################################################################################################

numpy_deformation = nib.load(result_registration["deformation"]).get_fdata()
numpy_moving = nib.load(path_moving).get_fdata()

result_warping_manual_numpy = registrationbaselines.displacement.deform_objects.deform_image_niftyreg_numpy(numpy_moving,
                                                                                                       numpy_deformation)

diff = TRUE_WARPED - result_warping_manual_numpy
print("\nLOAD WITH NUMPY AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# save warped
nib.save(nib.Nifti1Image(result_warping_manual_numpy,
                         nib_moving.affine), "registrationbaselines/tests/images/warped_numpy.nii.gz")

utils.print_histogram(
    torch.from_numpy(result_warping_manual_numpy), 10)

###########################################################################################################################################
# LOAD WITH TORCH AND DEFORM WITH NiftyReg
###########################################################################################################################################

torch_deformation = torch.from_numpy(
    nib.load(result_registration["deformation"]).get_fdata())
torch_moving = torch.from_numpy(
    nib.load(path_moving).get_fdata())
result_warping_manual_torch = registrationbaselines.displacement.deform_objects.deform_image_niftyreg_torch(torch_moving,
                                                                                                       torch_deformation)
diff = TRUE_WARPED - result_warping_manual_torch.cpu().numpy()
print("\nLOAD WITH TORCH AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# save warped
nib.save(nib.Nifti1Image(result_warping_manual_torch.cpu().numpy(),
                         nib_moving.affine), "registrationbaselines/tests/images/warped_torch.nii.gz")

utils.print_histogram(result_warping_manual_torch, 10)


###########################################################################################################################################
# remove files
#       - result_warping_manual_path
#       - result_registration["warped"]
#       - result_registration["deformation"]
###########################################################################################################################################


# Path(result_warping_manual_path).unlink()
# Path(result_registration["warped"]).unlink()
# Path(result_registration["deformation"]).unlink()


x = 0
