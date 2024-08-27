from pathlib import Path
import sys

import SimpleITK as sitk
import torch
import numpy as np


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.core import utils, utils_nifti  # nopep8


###########################################################################################################################################
# REGISTER WITH NiftyReg
###########################################################################################################################################

path_fixed = Path(
    "registrationbaselines/tests/images/fixed_x_11.nii.gz")
path_moving = Path(
    "registrationbaselines/tests/images/moving_x_11.nii.gz")
shape = sitk.GetArrayFromImage(
    sitk.ReadImage(path_fixed)).shape

# result_registration = utils.register_niftyreg(path_fixed, path_moving)

result_registration = {"warped": Path("registrationbaselines/tests/images/moving_x_11_warped.nii.gz"),
                       "deformation": Path("registrationbaselines/tests/images/deformation.nii.gz")}

NUMBER_OF_PIXELS = np.prod(shape)
TRUE_WARPED = sitk.GetArrayFromImage(
    sitk.ReadImage(result_registration["warped"]))
TRUE_WARPED_DEFORMATION = sitk.ReadImage(
    result_registration["deformation"])

###########################################################################################################################################
# LOAD WITH NiftyReg AND DEFORM WITH NiftyReg
###########################################################################################################################################
"""
result_warping_manual_path = utils.deform_image_niftyreg_path(path_moving,
                                                              result_registration["deformation"])
diff = TRUE_WARPED - \
    sitk.GetArrayFromImage(sitk.ReadImage(result_warping_manual_path))

print("\nLOAD WITH NiftyReg AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# rename result_warping_manual_path to "warped_niftyreg.nii.gz"
Path(result_warping_manual_path).rename(
    "registrationbaselines/tests/images/warped_niftyreg.nii.gz")
result_warping_manual_path = Path(
    "registrationbaselines/tests/images/warped_niftyreg.nii.gz")

utils.print_histogram(torch.from_numpy(sitk.GetArrayFromImage(
    sitk.ReadImage(result_warping_manual_path))), 10)
"""
###########################################################################################################################################
# LOAD WITH SITK AND DEFORM WITH NiftyReg
###########################################################################################################################################

sitk_deformation = sitk.ReadImage(result_registration["deformation"])
sitk_deformation_header1 = utils.get_sitk_header(sitk_deformation)
sitk.WriteImage(sitk_deformation, "deformation1.nii.gz")

sitk_deformation2 = sitk.ReadImage("deformation1.nii.gz")
sitk_deformation_header2 = utils.get_sitk_header(sitk_deformation2)

res12 = utils.compare_sitk_headers(sitk_deformation_header1,
                                      sitk_deformation_header2)

for a in res12:
    print(a)

sitk_moving = sitk.ReadImage(path_moving)

result_warping_manual_sitk = utils.deform_image_niftyreg_sitk(sitk_moving,
                                                              sitk_deformation,
                                                              path_moving,
                                                              # result_registration["deformation"]
                                                              )

diff = TRUE_WARPED - sitk.GetArrayFromImage(result_warping_manual_sitk)
print("\nLOAD WITH SITK AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# save warped
sitk.WriteImage(result_warping_manual_sitk,
                "registrationbaselines/tests/images/warped_sitk.nii.gz")

utils.print_histogram(torch.from_numpy(
    sitk.GetArrayFromImage(result_warping_manual_sitk)), 10)

###########################################################################################################################################
# LOAD WITH NUMPY AND DEFORM WITH NiftyReg
###########################################################################################################################################

numpy_deformation = sitk.GetArrayFromImage(
    sitk.ReadImage(result_registration["deformation"]))
numpy_moving = sitk.GetArrayFromImage(
    sitk.ReadImage(path_moving))

result_warping_manual_numpy = utils.deform_image_niftyreg_numpy(numpy_moving,
                                                                numpy_deformation,
                                                                metadata_sitk_deformation,
                                                                metadata_sitk_image)

diff = TRUE_WARPED - result_warping_manual_numpy
print("\nLOAD WITH NUMPY AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# save warped
sitk.WriteImage(sitk.GetImageFromArray(
    result_warping_manual_numpy), "registrationbaselines/tests/images/warped_numpy.nii.gz")

utils.print_histogram(torch.from_numpy(result_warping_manual_numpy), 10)

###########################################################################################################################################
# LOAD WITH TORCH AND DEFORM WITH NiftyReg
###########################################################################################################################################

torch_deformation = torch.from_numpy(
    sitk.GetArrayFromImage(sitk.ReadImage(result_registration["deformation"])))
torch_moving = torch.from_numpy(
    sitk.GetArrayFromImage(sitk.ReadImage(path_moving)))
result_warping_manual_torch = utils.deform_image_niftyreg_torch(torch_moving,
                                                                torch_deformation)
diff = TRUE_WARPED - result_warping_manual_torch.cpu().numpy()
print("\nLOAD WITH TORCH AND DEFORM WITH NiftyReg")
print(np.sum(np.abs(diff))/NUMBER_OF_PIXELS)

# save warped
sitk.WriteImage(sitk.GetImageFromArray(
    result_warping_manual_torch.cpu().numpy()), "registrationbaselines/tests/images/warped_torch.nii.gz")

utils.print_histogram(result_warping_manual_torch, 10)

###########################################################################################################################################
# Compare direction etc
###########################################################################################################################################
# compare direction etc
assert np.allclose(sitk_deformation.GetDirection(),
                   TRUE_WARPED_DEFORMATION.GetDirection())
assert np.allclose(sitk_deformation.GetOrigin(),
                   TRUE_WARPED_DEFORMATION.GetOrigin())
assert np.allclose(sitk_deformation.GetSpacing(),
                   TRUE_WARPED_DEFORMATION.GetSpacing())
assert np.allclose(sitk_deformation.GetSize(),
                   TRUE_WARPED_DEFORMATION.GetSize())


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
