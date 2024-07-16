from pathlib import Path

from typing import Union, Tuple, Any

import numpy as np
from scipy.ndimage import zoom
import ants.utils as utils_ants
import SimpleITK as sitk

from registrationbaselines.core.types import floatArray2Dor3D

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
                                      affine_matrix: floatArray2Dor3D,
                                      just_replace_existing_affine: bool = False
                                      ) -> sitk.Image:
    """
    Apply a transformation to a NIfTI image using a 4x4 matrix.

    @param path_image: Path to the NIfTI image file.
    @type path_image: Path

    @param affine_matrix: 4x4 affine transformation matrix.
    @type affine_matrix: np.ndarray

    @param just_replace_existing_affine: Flag indicating whether to replace the existing affine matrix 
                                         or to combine it with the new matrix.
    @type just_replace_existing_affine: bool

    @return: Transformed NIfTI image.
    @rtype: sitk.Image

    @raise AssertionError: If the affine matrix is not 4x4 or the last row is not [0, 0, 0, 1].
    @raise AssertionError: If the image file does not exist or is not a NIfTI file.
    """

    Warning("Haven't tested this functoin after the sitk rewrite")

    assert affine_matrix.shape == (
        4, 4), "The affine matrix must be a 4x4 matrix."
    assert np.allclose(affine_matrix[3], [
                       0, 0, 0, 1]), "The last row of the affine matrix must be [0, 0, 0, 1]."
    assert path_image.exists(), f"The image file {path_image} does not exist."
    assert path_image.suffix == ".nii" or path_image.suffixes == [
        ".nii", ".gz"], "The image file must be a NIfTI file."

    # Load the image
    image = sitk.ReadImage(str(path_image))

    # Get the current affine transform
    current_affine = np.array(image.GetDirection(),
                              dtype=np.float64).reshape((3, 3))
    current_affine = np.hstack(
        (current_affine, np.array(image.GetOrigin(), dtype=np.float64).reshape((3, 1))))
    current_affine = np.vstack((current_affine, [0, 0, 0, 1]))

    # Apply the transformation by updating or replacing the affine matrix
    if just_replace_existing_affine:
        new_affine = affine_matrix
    else:
        new_affine = np.dot(current_affine, affine_matrix)

    # Set the new affine transform
    new_direction = new_affine[:3, :3].flatten()
    new_origin = new_affine[:3, 3]
    image.SetDirection(new_direction)  # type: ignore
    image.SetOrigin(new_origin)  # type: ignore

    return image


def resample_nifti_image_isotropically(path_image: Path,
                                       which_dimension: str) -> sitk.Image:
    """
    Resample a NIfTI image isotropically based on the specified dimension.

    @param path_image: Path to the NIfTI image file.
    @type path_image: Path

    @param which_dimension: Specifies the dimension to use for isotropic resampling. 
                            Options are 'first', 'second', 'third', 'smallest', 'largest', or 'one'.
    @type which_dimension: str

    @return: Resampled NIfTI image.
    @rtype: sitk.Image

    @raise ValueError: If which_dimension is not one of the allowed values.
    """

    Warning("Haven't tested this functoin after the sitk rewrite")

    assert path_image.exists(), f"The image file {path_image} does not exist."
    assert path_image.suffix == ".nii" or path_image.suffixes == [
        ".nii", ".gz"], "The image file must be a NIfTI file."

    # Load the image
    image = sitk.ReadImage(str(path_image))

    # Get the current spacing
    spacing = image.GetSpacing()  # type: ignore
    assert isinstance(spacing, tuple), "spacing must be a tuple"
    spacing: Tuple[float, ...] = tuple(float(x) for x in spacing)

    # Determine new voxel size
    if which_dimension == "first":
        new_spacing = (spacing[0], spacing[0], spacing[0])
    elif which_dimension == "second":
        new_spacing = (spacing[1], spacing[1], spacing[1])
    elif which_dimension == "third":
        new_spacing = (spacing[2], spacing[2], spacing[2])
    elif which_dimension == "smallest":
        min_spacing = min(spacing)
        new_spacing = (min_spacing, min_spacing, min_spacing)
    elif which_dimension == "largest":
        max_spacing = max(spacing)
        new_spacing = (max_spacing, max_spacing, max_spacing)
    elif which_dimension == "one":
        new_spacing = (1.0, 1.0, 1.0)
    else:
        raise ValueError(
            "which_dimension must be 'first', 'second', 'third', 'smallest', 'largest', or 'one'.")

    # Define the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(new_spacing)

    original_size = np.array(image.GetSize(), dtype=int)
    new_size = (original_size * np.array(spacing) /
                np.array(new_spacing)).astype(int).tolist()
    resampler.SetSize(new_size)

    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    # Execute the resampling
    resampled_image: sitk.Image = resampler.Execute(image)

    return resampled_image


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
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Image

    assert path.exists(), f"File {path} does not exist."
    assert path.suffixes == ['.nii'] or path.suffixes == ['.nii', '.gz'], \
        f"File {path} is not a nifti file."
    assert intent_code in INTENT_CODES, \
        f"Intent code {intent_code} is not valid."

    image: Nifti1Image = nib.load(path)  # type: ignore

    # we have to construct a new image with the new intent code, otherwise we cannot overwrite
    header = image.header
    data: np.ndarray[Any, Any] = image.get_fdata()  # type: ignore

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
    assert path_fixed.exists() and path_h5.exists(), \
        f"Files {path_fixed} or {path_h5} do not exist."

    args = ['-d', '3',
            '-r', path_fixed.as_posix(),
            '-t', path_h5.as_posix(),
            '-o', f"[{path_niigz.as_posix()}, 1]",
            '--verbose']

    libfn = utils_ants.get_lib_fn('antsApplyTransforms')

    libfn(args)

    if not path_niigz.exists():
        raise FileNotFoundError("Couldn't transform.")
