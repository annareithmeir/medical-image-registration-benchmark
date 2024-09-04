import logging
from pathlib import Path

from typing import Any, Tuple, List

import numpy as np
import scipy.spatial
import SimpleITK as sitk
import torch

from registrationbaselines.core.types import floatArray2D, intArray1D, intArray2D, intArray3D


def turn_off_warnings() -> None:
    # Set the logging level for matplotlib to WARNING
    # matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    sitk.ProcessObject_SetGlobalWarningDisplay(False)


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
