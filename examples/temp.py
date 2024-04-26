from pathlib import Path
import nibabel as nib
import numpy as np

def pad_nifti_to_match(img1_path: Path, img2_path: Path) -> (nib.Nifti1Image, nib.Nifti1Image):
    """
    Load two NIfTI images from given paths, and pad the smaller image with zeros to match the dimensions
    of the larger one. Both images are then returned, potentially with one being a padded version.

    @param img1_path Path to the first NIfTI image.
    @param img2_path Path to the second NIfTI image.
    @return A tuple containing both NIfTI image objects, potentially with one padded to match the other.
    """
    # Load the NIfTI files
    img1 = nib.load(str(img1_path))
    img2 = nib.load(str(img2_path))
    
    # Extract data arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()
    
    # Determine new shape (maximum dimensions across both images)
    new_shape = np.maximum(data1.shape, data2.shape)
    
    # Initialize new data arrays if padding is needed
    if any(data1.shape != new_shape):
        new_data1 = np.zeros(new_shape, dtype=data1.dtype)
        new_data1[:data1.shape[0], :data1.shape[1], :data1.shape[2]] = data1
        padded_img1 = nib.Nifti1Image(new_data1, img1.affine)
    else:
        padded_img1 = img1
    
    if any(data2.shape != new_shape):
        new_data2 = np.zeros(new_shape, dtype=data2.dtype)
        new_data2[:data2.shape[0], :data2.shape[1], :data2.shape[2]] = data2
        padded_img2 = nib.Nifti1Image(new_data2, img2.affine)
    else:
        padded_img2 = img2
    
    return (padded_img1, padded_img2)

im1, im2 = pad_nifti_to_match(Path(r"/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz"),
                              Path(r"/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/affinely_registered_NiftyReg/tumor2_resampled111_normalized.nii.gz"))

# save both images
nib.save(im1, "padded_tumor1.nii.gz")
nib.save(im2, "padded_tumor2.nii.gz")