import pandas as pd
from pathlib import Path

from typing import Any, Tuple, Dict, Union, List, Optional

import yaml
import numpy as np
import SimpleITK as sitk
import torch


from registrationbaselines.core.types import floatArray2D, floarArray4Dor5D, array2Dor3D, intArray1D, intArray2D, intArray3D
from registrationbaselines.warping.utils_displacement import unit_displacement_to_displacement
from registrationbaselines.warping.utils_displacement import displacement_to_unit_displacement
from registrationbaselines.warping.utils_displacement import reverse_axis


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


def transform_csv(input_csv: Path, output_csv: Path) -> None:
    """
    Transform to a csv that can be read by Slicer
    """

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
