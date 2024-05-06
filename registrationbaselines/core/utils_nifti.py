from pathlib import Path
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import ants.utils as utils_ants

# intent codes for nifti files - at the moment we only need NIFTI_INTENT_DISPVECT for setting the displacement field
INTENT_CODES = ['NIFTI_INTENT_CORREL', 'NIFTI_INTENT_TTEST', 'NIFTI_INTENT_FTEST',
                'NIFTI_INTENT_ZSCORE', 'NIFTI_INTENT_CHISQ', 'NIFTI_INTENT_BETA',
                'NIFTI_INTENT_BINOM', 'NIFTI_INTENT_GAMMA', 'NIFTI_INTENT_POISSON',
                'NIFTI_INTENT_NORMAL', 'NIFTI_INTENT_FTEST_NONC', 'NIFTI_INTENT_CHISQ_NONC',
                'NIFTI_INTENT_LOGISTIC', 'NIFTI_INTENT_LAPLACE', 'NIFTI_INTENT_UNIFORM',
                'NIFTI_INTENT_TTEST_NONC', 'NIFTI_INTENT_WEIBULL', 'NIFTI_INTENT_CHI',
                'NIFTI_INTENT_INVGAUSS', 'NIFTI_INTENT_EXTVAL', 'NIFTI_INTENT_PVAL',
                'NIFTI_INTENT_LOGPVAL', 'NIFTI_INTENT_LOG10PVAL', 'NIFTI_FIRST_STATCODE',
                'NIFTI_LAST_STATCODE', 'NIFTI_INTENT_ESTIMATE', 'NIFTI_INTENT_LABEL',
                'NIFTI_INTENT_NEURONAME', 'NIFTI_INTENT_GENMATRIX', 'NIFTI_INTENT_SYMMATRIX',
                'NIFTI_INTENT_DISPVECT', 'NIFTI_INTENT_VECTOR', 'NIFTI_INTENT_POINTSET',
                'NIFTI_INTENT_TRIANGLE', 'NIFTI_INTENT_QUATERNION', 'NIFTI_INTENT_DIMLESS']


def transform_nifti_image_with_matrix(path_image: Path,
                                      affine_matrix: np.ndarray,
                                      just_replace_existing_affine: bool = False
                                      ) -> nib.Nifti1Image:
    """
    Apply a transformation to a NIfTI image using a 4x4 matrix.
    """

    assert affine_matrix.shape == (
        4, 4), "The rotation matrix must be a 4x4 matrix."
    assert np.allclose(affine_matrix[3], [
                       0, 0, 0, 1]), "The last row of the affine matrix must be [0, 0, 0, 1]."
    assert path_image.exists(), f"The image file {path_image} does not exist."
    assert path_image.suffix == ".nii" or path_image.suffixes == [
        ".nii", ".gz"], "The image file must be a NIfTI file."

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


def resample_nifti_image_isotropically(path_image: Path,
                                       which_dimension: str) -> nib.Nifti1Image:
    """
    Resample an image isotropically. If same_as_first_dimension is True,
    the new voxel size will be the same as the first, else it will be 1x1x1.
    """

    image = nib.load(path_image)
    affine = image.affine
    data = image.get_fdata()
    voxel_size = image.header.get_zooms()

    if which_dimension == "first":
        new_voxel_size = (voxel_size[0], voxel_size[0], voxel_size[0])
    elif which_dimension == "second":
        new_voxel_size = (voxel_size[1], voxel_size[1], voxel_size[1])
    elif which_dimension == "third":
        new_voxel_size = (voxel_size[2], voxel_size[2], voxel_size[2])
    elif which_dimension == "smallest":
        new_voxel_size = (min(voxel_size), min(voxel_size), min(voxel_size))
    elif which_dimension == "largest":
        new_voxel_size = (max(voxel_size), max(voxel_size), max(voxel_size))
    elif which_dimension == "one":
        new_voxel_size = (1, 1, 1)
    else:
        raise ValueError(
            "which_dimension must be 'first', 'second', 'third', 'smallest', 'largest', or 'one'.")

    # Calculate dimensions of the new volume
    # Get the voxel dimensions from the original affine
    voxel_dims = np.sqrt((affine * affine).sum(axis=0))[:-1]
    zoom_factors = voxel_dims / new_voxel_size

    # Calculate new data dimensions
    new_data_shape = (data.shape * zoom_factors).round().astype(int)

    # Resample the data
    resampled_data = zoom(data, zoom_factors, order=3)  # Cubic interpolation

    # Correct size discrepancy if necessary (due to rounding during zoom)
    resampled_data = np.pad(resampled_data,
                            [(0, max(0, new_data_shape[i] - resampled_data.shape[i]))
                             for i in range(3)],
                            mode='constant',
                            constant_values=0)

    # Create an identity affine
    identity_affine = np.eye(4)

    # Create a new NIfTI image with identity affine
    new_image = nib.Nifti1Image(resampled_data, identity_affine)

    return new_image


def set_intent_code(path: Path, intent_code: str) -> None:
    """
    Set the intent code of a nifti file. This is necessary when using a displacement field from NiftyReg or when we 
    build one frome scratch (NiftyReg set a custom displacement field that does not conform with the NIFTI standard).
    For this application we need to add NIFTI_INTENT_DISPVECT.

    Parameters
    ----------
    path : Path
        Path to the nifti file.
    intent_code : str
        The intent code to set.
    """

    assert path.exists(), f"File {path} does not exist."
    assert path.suffixes == ['.nii'] or path.suffixes == ['.nii', '.gz'], \
        f"File {path} is not a nifti file."
    assert intent_code in INTENT_CODES, \
        f"Intent code {intent_code} is not valid."

    image = nib.load(path)

    # we have to construct a new image with the new intent code, otherwise we cannot overwrite
    header = image.header
    data = image.get_fdata()

    # set the code
    header.set_intent(nib.nifti1.intent_codes[intent_code])

    # create new image
    new_image = nib.Nifti1Image(data, image.affine, header)

    # save the new image
    try:
        nib.save(new_image, path.as_posix())
    except Exception as e:
        print(f"Error saving the file: {e}")


def convert_h5_to_nii(path_fixed: Path, path_h5: Path, path_niigz: Path) -> None:
    """
    Convert the .h5 transformation to a .nii.gz transformation. Also requires the fixed (reference) image.
    """

    assert path_niigz.suffixes == ['.nii'] or path_niigz.suffixes == ['.nii', '.gz'], \
        f"File {path_niigz} is not a nifti file."
    assert path_h5.suffix == '.h5', f"File {path_h5} is not a .h5 file."
    assert path_fixed.exists() and path_h5.exists() and path_niigz.exists(), \
        f"Files {path_fixed}, {path_h5}, or {path_niigz} do not exist."

    args = ['-d', '3',
            '-r', path_fixed.as_posix(),
            '-t', path_h5.as_posix(),
            '-o', f"[{path_niigz.as_posix()}, 1]",
            '--verbose']

    libfn = utils_ants.get_lib_fn('antsApplyTransforms')

    libfn(args)

    if not path_niigz.exists():
        raise FileNotFoundError(f"Couldn't transform.")
