from pathlib import Path

import numpy as np
import nibabel as nib


def transform_nifti_image_with_matrix(path_image: Path,
                                      affine_matrix: np.ndarray,
                                      just_replace_existing_affine: bool = False
                                      ) -> nib.Nifti1Image:
    """
    Apply a transformation to a NIfTI image using a 4x4 matrix.
    """
    
    assert affine_matrix.shape == (4, 4), "The rotation matrix must be a 4x4 matrix."
    assert np.allclose(affine_matrix[3], [0, 0, 0, 1]),"The last row of the affine matrix must be [0, 0, 0, 1]."
    assert path_image.exists(), f"The image file {path_image} does not exist."
    assert path_image.suffix == ".nii" or path_image.suffixes == [".nii", ".gz"], "The image file must be a NIfTI file."
    
    # Load the image
    image = nib.load(path_image)
    data = image.get_fdata()
    
    # Apply the transformation by updating or replacing the affine matrix
    if just_replace_existing_affine:
        new_affine = affine_matrix
    else:
        new_affine = np.dot(image.affine, affine_matrix)

    # Create a new NIfTI image with the updated affine matrix
    new_image = nib.Nifti1Image(data, affine=new_affine)

    return new_image
