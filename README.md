# registrationBaselines


### NiftyReg
If you want to visualise the displacement field from NiftyReg in 3D Slicer you have to set the correct intent code (we have a utils for this: set_intent_code in utils_niftyreg) - but if you set this code, NiftyReg cannot use the file anymore.

### corrField
For some reason the resulting warped image is rotated around the x-axis by 180 degrees, so this has to be undone.
It also needs a mask to where to restrict the kyepoint search - in our implementation, if no mask is provided we create a temporary dummy mask with all 1s.
Also, both images and the mask need to be resampled to isotropic 1x1x1 spacing.