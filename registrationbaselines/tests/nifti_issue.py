import SimpleITK as sitk
import numpy as np

path = "displacement_field.nii.gz"
image_array = np.zeros((50, 60, 70, 3))
image_sitk = sitk.GetImageFromArray(image_array, isVector=True)

# Set the metadata key
image_sitk.SetMetaData("intent_name", "NREG_TRANS")
image_sitk.SetMetaData("intent_p1", "1")

# Write the image
sitk.WriteImage(image_sitk, path)

# check that it worked
image_test = sitk.ReadImage(path)
if image_test.GetMetaData("intent_name") != "NREG_TRANS":
    print("Metadata key intent_name was not set to NREG_TRANS.")
if image_test.GetMetaData("intent_p1") != "1":
    print("Metadata key intent_p1 was not set to 1.")
