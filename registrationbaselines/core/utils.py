import pandas as pd
from pathlib import Path

from typing import Any, Tuple, Dict, Union, List, Optional

from scipy.ndimage import map_coordinates
import yaml
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch
import torch.nn.functional as F


from registrationbaselines.core.types import floatArray2D, floarArray4Dor5D, array2Dor3D, intArray1D, intArray2D, intArray3D


def is_nifti(path: Path) -> None:
    """
    Function to check if a file is a nifti and exists.

    @param path: The path to the file.

    @return: True if the file is a nifti and exists, False otherwise.

    @raise ValueError: If the file is not a nifti.
    @raise FileNotFoundError: If the file does not exist
    """

    # check that file is .nii or .nii.gz
    suffixes = path.suffixes
    if not suffixes == ['.nii'] and not suffixes == ['.nii', '.gz']:
        raise ValueError(
            "The file should be in .nii or .nii.gz format, but it is {suffixes}.")

    # check that file exists
    if not path.exists():
        raise FileNotFoundError(
            f"File {path} does not exist.")


def is_isotropic(image: sitk.Image) -> None:
    """
    Checks if an image has isotropic voxel size.

    @param image: The image.

    @raise ValueError: If the voxel size is not isotropic.
    """

    spacing = np.array(image.GetSpacing(), np.float64)

    if len(spacing) == 3:
        if not np.allclose(spacing, spacing[0]):
            raise ValueError(
                f"Voxel size is not isotropic: {tuple(spacing)}")
    elif len(spacing) == 5:
        if not np.allclose(spacing[:3], spacing[0]) and not np.all(spacing[3:] == [1.0, 1.0]):
            raise ValueError(
                f"Voxel size is not isotropic: {tuple(spacing)}")
    else:
        raise ValueError(
            f"Spacing has to be elngth 3 (image/segmentation) or 5 (displacement)")


def is_direction_identity(image: sitk.Image) -> None:

    dimension = int(image.GetDimension())
    direction = np.abs(np.array(image.GetDirection(), np.float64))

    identity: np.ndarray[Any, np.dtype[np.float64]]
    match dimension:
        case 2:
            identity = np.eye(2)
        case 3:
            identity = np.eye(3)
        case 4:
            identity = np.eye(4)
        case _:
            identity = np.eye(5)

    if not np.allclose(direction, identity.flatten()):
        raise ValueError(
            f"Direction is not identity: {tuple(direction)}"
        )


def are_offdiagonal_direction_elements_zero(image: sitk.Image) -> None:

    dir = np.abs(np.array(image.GetDirection(), np.float64))

    if len(dir) == 9:
        dir_offdiagonal = list(dir[1:4]) + list(dir[5:8])
        zeros = [0.0 for _ in range(6)]
    elif len(dir) == 25:
        dir_offdiagonal = list(dir[1:6]) + list(dir[7:12]) + \
            list(dir[13:18]) + list(dir[19:24])
        zeros = [0.0 for _ in range(20)]
    else:
        raise ValueError(
            f"Direction has to be elngth 9 (image/segmentation) or 25 (displacement)")

    if dir_offdiagonal != zeros:
        raise ValueError("Off-diagonal elements are not zero")


def flip(x: torch.Tensor, dim: int):
    """
    Flip order of a specific dimension dim

    x (Tensor): input tensor
    dim (int): axis which should be flipped
    return (Tensor): returns the tensor with the specified axis flipped
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def reverse_axis(image: torch.Tensor) -> torch.Tensor:
    """
    Flips the order of the axis representing the space dimensions (preceeding dimensions are ignored).
    Respectively, the axis holding the vectors is flipped as well

    Note: the method is inplace
    """
    # reverse order of axis to follow the convention of SimpleITK
    order = list(reversed(range(image.ndim-1)))
    order.append(len(order))
    image = image.squeeze_().permute(tuple(order))
    image = flip(image, image.ndim-1)

    return image


def load_displacement(path: Path) -> torch.Tensor:
    """
    Load a displacement field from a file and return it as a torch tensor in the shape H,W,D,3

    @param path: The path to the displacement field file.

    @return: The displacement field.

    @raise ValueError: If the intent code of the displacement field is not NIFTI_INTENT_DISPVECT.
    @raise ValueError: If the displacement field has wrong dimensions.
    """

    is_nifti(path)

    displacement_sitk: sitk.Image = sitk.ReadImage(path)

    is_isotropic(displacement_sitk)
    is_direction_identity(displacement_sitk)
    are_offdiagonal_direction_elements_zero(displacement_sitk)

    displacement_array: floarArray4Dor5D = sitk.GetArrayFromImage(
        displacement_sitk)

    # check intent code
    # 1006 = NIFTI_INTENT_DISPVECT
    if displacement_sitk.GetMetaData("intent_code") != "1006":
        raise ValueError(
            "The intent code of the displacement field should be NIFTI_INTENT_DISPVECT.")

    # check dimensions
    shape = displacement_array.shape

    # check that it is 5D
    if len(shape) != 5:
        raise ValueError(
            f"Dimension is not 5D: {len(shape)}"
        )

    separating_dimension_correct = shape[1] == 1  # dim 1 is dummy
    # dim 0 is vector dimension, which has to correspond to spatial dimensions
    vector_dimension_correct = shape[0] == len(shape) - 2

    if not separating_dimension_correct or not vector_dimension_correct:
        raise ValueError(
            "The displacement field should have spatial dimensions as the last dimensions \
                and a vector dimension as the first dimension and separated by a dummy dimension.")

    displacement_tensor = torch.from_numpy(displacement_array)
    if displacement_tensor.dtype != torch.float32:
        raise TypeError(
            f"Dsiplacement is not torch.float32: {displacement_tensor.dtype}"
        )

    # remove separating dummy dimension
    displacement_tensor = displacement_tensor.squeeze()

    # move the vector dimension to the last dimension
    new_order = list(range(1, displacement_tensor.dim())) + [0]
    displacement_tensor = displacement_tensor.permute(new_order)

    # should be unit displacement
    displacement_tensor = displacement_to_unit_displacement(
        displacement_tensor)

    displacement_tensor = reverse_axis(displacement_tensor)

    return displacement_tensor


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load a nifti image from a file and return it as a numpy array.

    The file should be in .nii or .nii.gz format.
    The voxel size should be isotropic.
    The direction should be identity.
    The image should be 2D, 3D or 4D.
    The image should be float or integer.

    @param image_path: The path to the image file.
    @type image_path: Path

    @return: The image.
    @rtype: floatArray2Dor3Dor4D
    """

    is_nifti(image_path)

    image_sitk: sitk.Image = sitk.ReadImage(image_path)

    is_isotropic(image_sitk)
    is_direction_identity(image_sitk)
    are_offdiagonal_direction_elements_zero(image_sitk)

    image_array: array2Dor3D = sitk.GetArrayFromImage(image_sitk)

    dimension = image_array.ndim

    # check that it 3D
    if dimension != 3:
        raise ValueError(
            f"Dimension of {image_path} is not 3D: {dimension}"
        )

    return_tensor = torch.from_numpy(image_array).squeeze()
    # check that image is float or int
    if not (return_tensor.dtype == torch.uint8 or return_tensor.dtype == torch.float32):
        raise TypeError(
            f"Image is not float or int: {return_tensor.dtype}"
        )

    return_tensor = return_tensor.permute(2, 1, 0)

    return return_tensor


def get_image_spacing(image_path: Path) -> Tuple[float, float, float]:
    """
    Get the spacing of an image.

    @param image_path: The path to the image file.
    @type image_path: Path

    @return: The spacing.
    @rtype: Tuple[float, float, float]
    """

    is_nifti(image_path)

    image_sitk: sitk.Image = sitk.ReadImage(image_path)

    spacing = image_sitk.GetSpacing()

    if len(spacing) != 3:
        raise ValueError(f"Spacing is not 3D: {spacing}")

    return spacing[2], spacing[1], spacing[1]


def load_keypoints(keypoints_path: Path) -> torch.Tensor:
    """
    Load keypoints from a csv file to a torch tensor of shape [N,3].

    dtype: float32

    The keypoints have to be in x,y,z order and they will be converted to z,y,x.
    The switch happens because extracting a np array from an sitk image does that too.

    @param keypoints_path: Path to the .txt keypoints file
    @return: Keypoints as a torch tensor of shape [N,3].
    """

    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")

    keypoints = np.loadtxt(keypoints_path, delimiter=',', dtype=np.float32)
    keypoints = torch.from_numpy(keypoints)

    if keypoints.ndim != 2:  # this is enforced by numpy, but let's keep it here
        raise ValueError(f"Keypoints should be 2D: {keypoints.ndim}")
    if keypoints.shape[1] != 3:
        raise ValueError(
            f"Keypoints should have 3 columns: {keypoints.shape[1]}")

    # switch x and z
    # keypoints[:, [0, 2]] = keypoints[:, [2, 0]]

    return keypoints


def save_image(image: torch.Tensor, image_path: Path, spacing: Tuple[float, ...]) -> None:
    """
    Save a numpy array as a nifti image.

    The voxel size will be isotropic.
    The direction will be identity.
    The image can be 2D, 3D or 4D.
    """

    # check that file is .nii or .nii.gz
    if not image_path.suffix == '.nii' and not image_path.suffixes == ['.nii', '.gz']:
        raise ValueError(
            "The path should be in .nii or .nii.gz format.")

    # check that it is 3D
    if image.ndim != 3:
        raise ValueError(
            f"Dimension of image is not 3D: {image.ndim}"
        )

    # spacing has to match the image
    if len(spacing) != image.ndim:
        raise ValueError(
            "The spacing does not match the image dimensions."
        )

    image = image.permute(2, 1, 0)
    spacing = list(reversed(spacing))

    sitk_image = sitk.GetImageFromArray(image.detach().cpu().numpy())
    sitk_image.SetSpacing(spacing)

    sitk.WriteImage(sitk_image, image_path)

    # check that file was written
    if not image_path.exists():
        raise FileNotFoundError(f"File {image_path} was not written.")


def save_displacement(displacement: torch.Tensor,
                      image_path: Path,
                      spacing: Tuple[float, ...]) -> None:
    """
    Save a displacement field as a nifti image.

    The voxel size will be isotropic.
    The direction will be identity.

    BUGFIX_0: we have to reverse the axis of the displacement (and in the spacing),
              to match the reversal in loading
    """
    from registrationbaselines.core import utils_nifti

    # check that file is .nii or .nii.gz
    if not image_path.suffix == '.nii' and not image_path.suffixes == ['.nii', '.gz']:
        raise ValueError(
            "The path should be in .nii or .nii.gz format.")

    # check dimensions
    shape = displacement.shape

    # check that it is 4D
    if len(shape) != 4:
        raise ValueError(f"Dimension of displacement is not 4D: {len(shape)}")
    if shape[-1] != 3:
        raise ValueError(
            "The displacement field should have the vector dimension as the last dimension.")

    if displacement.dtype != torch.float32:
        raise TypeError(
            f"Dsiplacement is not torch.float32: {displacement.dtype}"
        )

    # spacing has to match the image
    if len(spacing) != len(shape):
        raise ValueError(
            "The spacing does not match the image dimensions."
        )

    # BUGFIX_0
    displacement = reverse_axis(displacement)

    # should be unit displacement
    displacement = unit_displacement_to_displacement(displacement)

    # move vector dimension from back to front
    displacement = displacement.permute(3, 0, 1, 2)
    spacing = (spacing[-1],) + spacing[:-1]

    # insert separating dimension
    displacement = displacement.unsqueeze(1)

    # add dummy spacing
    spacing = (spacing[0], 1) + spacing[1:]

    # BUGFIX_0
    # reverse spacing
    spacing = (spacing[4], spacing[3], spacing[2], spacing[1], spacing[0])

    sitk_displacement = sitk.GetImageFromArray(
        displacement.detach().cpu().numpy())
    sitk_displacement.SetSpacing(spacing)

    sitk.WriteImage(sitk_displacement, image_path)

    utils_nifti.set_intent_code(image_path, 'NIFTI_INTENT_DISPVECT')


def get_affine_from_image(image: sitk.Image) -> floatArray2D:
    """
    Get the affine matrix from a SimpleITK image.

    @param image: The SimpleITK image.
    @type image: sitk.Image

    @return: The affine matrix.
    @rtype: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]
    """

    if image.GetDimension() != 3:
        raise ValueError("The image should be 3D.")

    direction = np.array(image.GetDirection(), np.float64).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), np.float64)
    origin = np.array(image.GetOrigin(), np.float64)

    # Construct the affine matrix
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction * spacing[:, None]
    affine[:3, 3] = origin

    return affine


def read_config(file_path: Path) -> dict[str, Any]:
    """
    Read the configuration file.

    @param file_path: The path to the configuration file.
    @rtype file_path: Path

    @return: The configuration.
    @rtype: dict[str, Any]
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def save_array_to_nii_gz_image(array: np.ndarray, filename: Path, affine: np.ndarray = None) -> None:
    """
    Saves a 3D numpy array to a .nii.gz file
    @param array: array
    @param filename: filename for daving
    @param affine: affine matrix of shaoe (4,4)
    @return:
    """
    assert array.ndim == 3
    image = sitk.GetImageFromArray(array)

    if affine is not None:
        # SimpleITK uses the direction cosine matrix, origin, and spacing to set the affine
        direction = affine[:3, :3].flatten()
        origin = affine[:3, 3]
        spacing = np.linalg.norm(affine[:3, :3], axis=0)

        image.SetDirection(direction)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

    sitk.WriteImage(image, str(filename))


def get_sitk_header(sitk_image: sitk.Image) -> dict[str, Any]:

    result_dict = {}

    for key in sitk_image.GetMetaDataKeys():
        value = sitk_image.GetMetaData(key)
        result_dict[key] = value

    return result_dict


def compare_sitk_headers(header1: dict[str, Any], header2: dict[str, Any]) -> List[Any]:

    differences = []

    for key in header1.keys():
        if key not in header2.keys():
            differences.append(f"Key {key} not in header2")
        else:
            if header1[key] != header2[key]:
                differences.append(
                    f"Key {key} has different values: header1({header1[key]}) vs header2({header2[key]})")

    for key in header2.keys():
        if key not in header1.keys():
            differences.append(f"Key {key} not in header1")
        else:
            if header1[key] != header2[key]:
                differences.append(
                    f"Key {key} has different values: header1({header1[key]}) vs header2({header2[key]})")

    return differences


def save_array_to_nii_gz_displacement_field(array: np.ndarray, filename: Path, affine: np.ndarray = None) -> None:
    """
    Saves a displacement field in form of np array to a .nii.gz file
    @param array: np array of shape (H,W,D,3)
    @param filename: filename for saving
    @param affine: affine matrix of shape (4,4)
    @return:
    """
    assert array.ndim == 4
    assert array.shape[-1] == 3
    image = sitk.GetImageFromArray(array, isVector=True)

    if affine is not None:
        # SimpleITK uses the direction cosine matrix, origin, and spacing to set the affine
        direction = affine[:3, :3].flatten()
        origin = affine[:3, 3]
        spacing = np.linalg.norm(affine[:3, :3], axis=0)

        image.SetDirection(direction)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

    sitk.WriteImage(image, str(filename))


def displacement_to_unit_displacement(displacement: torch.Tensor) -> torch.Tensor:
    """
    Convert a displacement field to a unit displacement field.
    The standard unit of displacement is a half-image, so a displacement vector of magnitude 2
    means that the displacement distance is equal to the side length of the displaced image.
    """

    disp = torch.zeros_like(displacement)

    for dim in range(displacement.shape[-1]):
        disp[..., dim] = 2.0 * displacement[..., dim] / \
            float(displacement.shape[-dim - 2] - 1)

    return disp


def unit_displacement_to_displacement(displacement: torch.Tensor) -> torch.Tensor:

    disp = torch.zeros_like(displacement)

    for dim in range(displacement.shape[-1]):
        disp[..., dim] = float(
            displacement.shape[-dim - 2] - 1) * displacement[..., dim] / 2.0

    return disp


def deform_image_niftyreg_path(path_image: Path,
                               path_deformation: Path) -> Path:

    from registrationbaselines.core import utils_commandline

    path_warped_image = Path(
        path_image.as_posix().replace(".nii", "_warpedManually.nii"))

    command = ["/u/home/koeglf/Documents/code/registrationbaselines/registrationbaselines/libraries/NiftyReg/reg_resample_ubuntu",
               '-ref', path_image.as_posix(),
               '-flo', path_image.as_posix(),
               '-trans', path_deformation.as_posix(),
               '-res', path_warped_image.as_posix()]

    utils_commandline.run_command_in_terminal(command,
                                              path_warped_image.exists,
                                              print_command_list=False)

    return path_warped_image


def deform_image_niftyreg_nibabel(image: nib.Nifti1Image,
                                  deformation: nib.Nifti1Image) -> nib.Nifti1Image:
    path_image = Path("image.nii.gz")
    nib.save(image, path_image)

    path_deformation = Path("deformation.nii.gz")
    nib.save(deformation, path_deformation)

    path_warped_image = deform_image_niftyreg_path(path_image,
                                                   path_deformation)

    warped = nib.load(path_warped_image)

    path_image.unlink()
    path_deformation.unlink()

    return warped


def deform_image_niftyreg_sitk(image: sitk.Image,
                               deformation: sitk.Image,
                               image_path: Optional[Path] = None,
                               deformation_path: Optional[Path] = None) -> sitk.Image:

    if image_path:
        path_image = image_path
    else:
        path_image = Path("image.nii.gz")
        sitk.WriteImage(image, path_image)

    if deformation_path:
        path_deformation = deformation_path
    else:
        path_deformation = Path("deformation.nii.gz")
        sitk.WriteImage(deformation, path_deformation)

    path_warped_image = deform_image_niftyreg_path(path_image,
                                                   path_deformation)

    warped = sitk.ReadImage(path_warped_image)

    if not image_path:
        path_image.unlink()
    if not deformation_path:
        path_deformation.unlink()

    return warped


def deform_image_niftyreg_numpy(image: np.ndarray[Any, Any],
                                deformation: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:

    image_nib = nib.Nifti1Image(image.astype(np.float32), affine=None)
    deformation_nib = nib.Nifti1Image(deformation, affine=None)

    warped_nib = deform_image_niftyreg_nibabel(image_nib, deformation_nib)

    warped = warped_nib.get_fdata(dtype=np.float32)

    return warped


"""
def deform_image_niftyreg_numpy(image: np.ndarray[Any, Any],
                                deformation: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:

    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    sitk_deformation = sitk.GetImageFromArray(deformation)

    sitk_warped = deform_image_niftyreg_sitk(sitk_image, sitk_deformation)

    warped = sitk.GetArrayFromImage(sitk_warped).astype(np.float32)

    return warped
"""

"""
def deform_image_niftyreg_torch(image: torch.Tensor,
                                deformation: torch.Tensor) -> torch.Tensor:

    sitk_image = sitk.GetImageFromArray(
        image.detach().cpu().numpy().astype(np.float32))
    sitk_deformation = sitk.GetImageFromArray(
        deformation.detach().cpu().numpy())

    sitk_warped = deform_image_niftyreg_sitk(sitk_image, sitk_deformation)

    warped = torch.from_numpy(
        sitk.GetArrayFromImage(sitk_warped).astype(np.float32))

    return warped
"""


def deform_image_niftyreg_torch(image: torch.Tensor,
                                deformation: torch.Tensor) -> torch.Tensor:

    image_np = image.detach().cpu().numpy().astype(np.float32)
    deformation_np = deformation.detach().cpu().numpy()

    image_nib = nib.Nifti1Image(image_np, affine=None)
    deformation_nib = nib.Nifti1Image(deformation_np, affine=None)

    warped_nib = deform_image_niftyreg_nibabel(image_nib, deformation_nib)

    warped = torch.from_numpy(warped_nib.get_fdata(dtype=np.float32))

    return warped


def register_niftyreg(path_fixed: Path,
                      path_moving: Path) -> Dict[str, Path]:

    from registrationbaselines.core import utils_commandline, utils_niftyreg

    # check that both images exist
    assert path_fixed.exists(
    ), f"File {path_fixed} does not exist."
    assert path_moving.exists(
    ), f"File {path_moving} does not exist."

    path_reg_f3d = Path(
        "/u/home/koeglf/Documents/code/registrationbaselines/registrationbaselines/libraries/NiftyReg/reg_f3d_ubuntu")

    path_result_deformed = Path(
        path_moving.as_posix().replace(".nii", "_warped.nii"))
    path_result_gird = path_moving.parent / "deformation_temp.nii.gz"

    command = [path_reg_f3d.as_posix(),
               '-ref', path_fixed.as_posix(),
               '-flo', path_moving.as_posix(),
               '-res', path_result_deformed.as_posix(),
               '-cpp', path_result_gird.as_posix()]

    utils_commandline.run_command_in_terminal(command,
                                              path_result_deformed.exists,
                                              print_command_list=False)

    path_result_deformation = utils_niftyreg.convert_transformation_to_displacement_field(
        path_result_gird, path_fixed)

    return {"warped": path_result_deformed,
            "deformation": path_result_deformation}


def print_histogram(tensor: torch.Tensor, bins: int) -> None:
    # Flatten the 3D tensor to 1D
    flattened_tensor = tensor.flatten()

    # Calculate the histogram with 10 bins between 0.0 and 1.0
    hist = torch.histc(flattened_tensor, bins=10, min=0.0, max=1.0)

    # Normalize the histogram counts to a reasonable scale for display
    max_count = hist.max().item()
    scale_factor = 200 / max_count if max_count > 0 else 1

    # Print the histogram with horizontal bars
    for i in range(bins):
        bin_start = i / float(bins)
        bar_length = hist[i].item() * scale_factor

        if 0.1 < bar_length < 1:
            bar_length = 1

        bar = '█' * int(bar_length)
        print(f"{bin_start:.1f} [{bar}]")


def compute_grid(image_size: torch.Size,
                 dtype: torch.dtype = torch.float32,
                 device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Compute a normalized grid for a given image size.

    @param image_size: The size of the image. Should be a list of two or three integers.
    @type image_size: List[int]

    @param dtype: The desired data type of the returned grid.
    @type dtype: torch.dtype

    @param device: The desired device of the returned grid.
    @type device: Union[str, torch.device]

    @return: A tensor representing the normalized grid.
    @rtype: torch.Tensor
    """

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        raise ValueError(f"Error: {dim} is not a valid grid dimension.")


def deform_image(image: torch.Tensor,
                 displacement: torch.Tensor) -> torch.Tensor:
    """
    Apply a deformation to an image using the provided deformation.
    If the image is of type uint8 ie a segmentation map,
    it automatically uses mode='nearest' and returns an image of type uint8
    @param image: if label map then dtype must be torch.uint8, else torch.float
    @param displacement:
    @return:
    """

    # squeeze both image and displacement to ensure we only have spatial dimensions
    image = image.squeeze()
    displacement = displacement.squeeze()

    # convert to unit displacement if range is not [-1,1]
    if displacement.min() < -1 or displacement.max() > 1:
        displacement = displacement_to_unit_displacement(displacement)

    if image.ndim != displacement.ndim - 1:
        raise ValueError(
            "The displacement field should have one more dimension than the image.")

    if image.dtype == torch.uint8:
        mode = 'nearest'
        image = image.float()
    elif image.dtype == torch.float32:
        mode = 'bilinear'
    else:
        raise ValueError(
            "The image should be either uint8 or float32.")

    grid = compute_grid(
        image.shape, dtype=image.dtype, device=image.device)

    # unsqueeze image and displacement to conform to grid_sample requirements
    image = image.unsqueeze(0).unsqueeze(0)
    displacement = displacement.unsqueeze(0)

    # warp image
    warped_image = F.grid_sample(
        image, displacement + grid, mode=mode, align_corners=True).squeeze()

    if warped_image.ndim != image.squeeze().ndim:
        raise ValueError(
            "The warped image should have the same number of dimensions as the original image. \
                Something wen wrong with deforming")

    if mode == 'nearest':
        warped_image = warped_image.to(dtype=torch.uint8)
    return warped_image.squeeze()


def explore_memory():
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2)} MB")

    # Print the amount of reserved memory
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Reserved memory: {reserved_memory / (1024 ** 2)} MB")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory - allocated_memory

    print(f"Available memory: {available_memory / (1024 ** 2)} MB")
    print(f"Available memory: {available_memory / total_memory * 100} %")


def rgb_to_grayscale(rgb_image):
    r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
    grayscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale_image


def normalize_tensor_to_0_1(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def deform_keypoints(moving_keypoints: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """
    Deforms keypoints according to the pull convention using grid_sample.

    @param moving_keypoints: Tensor of shape (N, 3) where N is the number of keypoints (18 in this case).
    @param displacement: Tensor of shape (201, 201, 201, 3) containing the displacement field.
    @return: Deformed keypoints as a Tensor of shape (N, 3).
    """

    if displacement.min() >= -1 and displacement.max() <= 1:
        displacement = unit_displacement_to_displacement(displacement)

    N = moving_keypoints.shape[0]

    moving_keypoints = moving_keypoints[:, [2, 1, 0]]

    # Normalize moving_keypoints to the range [-1, 1] for grid_sample
    grid = moving_keypoints.unsqueeze(0)  # Shape (1, N, 3)

    # Normalize the grid to [-1, 1] based on the displacement field size
    grid = (grid - torch.tensor([displacement.shape[0] / 2, displacement.shape[1] / 2, displacement.shape[2] / 2], device=grid.device)) \
        / torch.tensor([displacement.shape[0] / 2, displacement.shape[1] / 2, displacement.shape[2] / 2], device=grid.device)

    # Reshape the grid to the correct shape for grid_sample
    grid = grid.view(1, 1, 1, N, 3)  # Shape (1, 1, 1, N, 3)

    # Prepare displacement field for grid_sample
    displacement = displacement.permute(3, 0, 1, 2).unsqueeze(
        0)  # Shape (1, 3, 201, 201, 201)

    # Use grid_sample to sample the displacement field at the keypoints' locations
    sampled_displacement = F.grid_sample(
        displacement, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Reshape the sampled displacement to match the original keypoints shape
    sampled_displacement = sampled_displacement.squeeze().transpose(0, 1)  # Shape (N, 3)

    # Apply the displacement to the keypoints
    deformed_keypoints = moving_keypoints - sampled_displacement  # Pull convention

    deformed_keypoints = deformed_keypoints[:, [2, 1, 0]]

    return deformed_keypoints


def deform_keypointsOLD(moving_keypoints: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """
    Deforms keypoints according to the pull convention

    Map the moving keypoints to the fixed keypoints using the displacement field
    The displacement field should be pixel-based for this to work, so in case it is a unit-displacement field, it is first converted...
    @param moving_keypoints:
    @param displacement: of shape (...,3) and optimally non-unit displacement (will be converted otherwise)
    @return:
    """

    if displacement.min() >= -1 and displacement.max() <= 1:
        displacement = unit_displacement_to_displacement(displacement)

    # Transpose the keypoints to match the shape for map_coordinates (3, N)
    moving_keypoints_t = moving_keypoints.transpose(0, 1)

    if moving_keypoints.shape[-1] == 3:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, :, 0], moving_keypoints_t)
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, :, 1], moving_keypoints_t)
        mov_lms_disp_z = map_coordinates(
            displacement[:, :, :, 2], moving_keypoints_t)
        mov_lms_disp = torch.tensor(
            (mov_lms_disp_x, mov_lms_disp_y, mov_lms_disp_z)).transpose(0, 1)
    elif moving_keypoints.shape[-1] == 2:
        mov_lms_disp_x = map_coordinates(
            displacement[:, :, 0], moving_keypoints_t)
        mov_lms_disp_y = map_coordinates(
            displacement[:, :, 1], moving_keypoints_t)
        mov_lms_disp = torch.tensor(
            (mov_lms_disp_x, mov_lms_disp_y)).transpose(0, 1)
    else:
        raise ValueError(
            "The landmark shape is not supported. It should be either 2 or 3.")

    """
    ######################################################################################################
    ###### TEMPORARY  ################
    ######################################################################################################
    # Step 2: Get the integer coordinates of landmarks
    moving_coords = moving_keypoints.long()

    # Ensure the coordinates are within the valid range
    moving_coords[:, 0] = torch.clamp(
        moving_coords[:, 0], 0, displacement.shape[1] - 1)
    moving_coords[:, 1] = torch.clamp(
        moving_coords[:, 1], 0, displacement.shape[2] - 1)
    moving_coords[:, 2] = torch.clamp(
        moving_coords[:, 2], 0, displacement.shape[3] - 1)

    displacement = unit_displacement_to_displacement(displacement)

    # Step 3: Extract the displacements for each landmark
    displacements = displacement[moving_coords[:, 0],
                                 moving_coords[:, 1],
                                 moving_coords[:, 2], :]

    # Step 4: Apply the displacements
    displaced_landmarks_zyx = moving_coords.float() - displacements

    return displaced_landmarks_zyx
    """

    deformed_keypoints = moving_keypoints - mov_lms_disp  # pull
    return deformed_keypoints


def transform_csv(input_csv: Path, output_csv: Path):
    # Load the CSV file using np.genfromtxt
    coords = np.genfromtxt(input_csv, delimiter=',')

    # Prepare the data for the new format
    data = {
        'label': [f'F-{i+1}' for i in range(coords.shape[0])],
        'l': coords[:, 0],
        'p': coords[:, 1],
        's': coords[:, 2],
        'defined': [1] * coords.shape[0],
        'selected': [1] * coords.shape[0],
        'visible': [1] * coords.shape[0],
        'locked': [0] * coords.shape[0],
        'description': [''] * coords.shape[0]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to the output CSV file
    df.to_csv(output_csv, index=False)


def create_method_name_for_wandb(method_name: str, wandb_config: Dict[str, Union[str, int, float, bool]]) -> str:
    """
    Create the method name for wandb.
    """

    for key, value in wandb_config.items():

        if key not in ['result_path', 'method_name'] and 'path' not in key:
            if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
                method_name += f"___{key}_{str(value).lower()}"
            elif isinstance(value, list):
                method_name += f"___{key}_{value}"
            elif value is not None:
                beautified_param = value.replace('-', '').replace(' ', '_')
                method_name += f"___{key}_{beautified_param}"

        if len(method_name) > 100:
            break

    return method_name


def find_points_inside_convex_hull(points: intArray2D,
                                   hull: scipy.spatial.ConvexHull,
                                   image_shape: Tuple[int, ...]
                                   ) -> Tuple[intArray1D, intArray1D, intArray1D]:
    """
    deln = scipy.spatial.Delaunay(points[hull.vertices]):   This is where Delaunay triangulation comes in.
            It's applied to the vertices of the convex hull. Delaunay triangulation creates a triangulation
            of points such that no point is inside the circumcircle of any triangle.

    idx = np.stack(np.indices(image.shape), axis=-1): This creates an array of all possible coordinates in the
            image shape.

    out_idx = np.nonzero(deln.find_simplex(idx) + 1): deln.find_simplex(idx) checks which simplex (triangle in 2D,
            tetrahedron in 3D) each point in idx belongs to.
            Adding 1 and using np.nonzero() effectively finds all points that are inside the convex hull.

    @param points: The points inside the convex hull.
    @param hull: The convex hull.
    @param image_shape: The shape of the image.
    @return: The indices of the points inside the convex hull.
    """

    deln = scipy.spatial.Delaunay(points[hull.vertices])

    idx = np.stack(np.indices(image_shape), axis=-1)

    out_idx = np.nonzero(deln.find_simplex(idx) + 1)

    return out_idx


def get_convex_hull_mask(image: intArray3D) -> intArray3D:
    """
    Creates a mask from the convex hull. All values outside of the hull are set to 0.

    Usefull for when e.g. segmentations in the deformed image are outside of the FOV of the fixed image.

    ToDo - what does Delaunay do?

    @param image: The image.
    @return: The mask.
    """
    points = np.transpose(np.where(image))

    hull = scipy.spatial.ConvexHull(points)

    out_idx = find_points_inside_convex_hull(points,
                                             hull,
                                             image.shape)

    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1

    out_img = out_img.astype(np.uint8)

    return out_img
