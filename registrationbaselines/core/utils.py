import logging
from pathlib import Path

from typing import Any, Tuple, List

import numpy as np
import scipy.spatial
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from registrationbaselines.core.types import floatArray2D, intArray1D, intArray2D


def turn_off_warnings() -> None:
    # Set the logging level for matplotlib to WARNING
    # matplotlib.use('Agg')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    sitk.ProcessObject_SetGlobalWarningDisplay(False)


def is_affine_identity(affine: floatArray2D) -> None:
    """
    Check if an affine matrix is the identity matrix.

    @param affine: The affine matrix.
    @type affine: np.ndarray[Tuple[int, int], np.dtype[np.float64]]

    @raise ValueError: If the affine matrix is not the identity matrix.
    """

    identity = np.eye(4, dtype=np.float64)

    if not np.allclose(affine, identity):
        raise ValueError("Affine matrix is not identity.")


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


def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
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


def get_convex_hull_mask(image: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask from the convex hull. All values outside of the hull are set to 0.

    Usefull for when e.g. segmentations in the deformed image are outside of the FOV of the fixed image.

    ToDo - what does Delaunay do?

    @param image: The image.
    @return: The mask.
    """

    image_array = image.detach().cpu().numpy()

    points = np.transpose(np.where(image_array))

    hull = scipy.spatial.ConvexHull(points)

    out_idx = find_points_inside_convex_hull(points,
                                             hull,
                                             image_array.shape)

    out_img = np.zeros(image_array.shape)
    out_img[out_idx] = 1

    result = torch.from_numpy(out_img).to(torch.uint8).to(image.device)

    return result


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


def reshape_tensor(tensor: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
    """
    Reshape a tensor by padding or cropping each dimension as needed.

    Args:
    tensor (torch.Tensor): The input tensor to reshape
    new_shape (tuple): The desired output shape

    Returns:
    torch.Tensor: The reshaped tensor
    """

    # Ensure the new_shape has the same number of dimensions as the input tensor
    assert len(new_shape) == tensor.dim(
    ), "New shape must have the same number of dimensions as the input tensor"

    current_shape = tensor.shape

    # Initialize the pad and crop parameters
    pad_sizes = []
    crop_slices = []

    for i, (current, new) in enumerate(zip(current_shape, new_shape)):
        if new > current:
            # Padding needed
            pad_left = (new - current) // 2
            pad_right = new - current - pad_left
            pad_sizes.extend([pad_left, pad_right])
        elif new < current:
            # Cropping needed
            crop_start = (current - new) // 2
            crop_end = crop_start + new
            crop_slices.append(slice(crop_start, crop_end))
        else:
            # No change needed
            pad_sizes.extend([0, 0])
            crop_slices.append(slice(None))

    # Reverse pad_sizes because F.pad expects them in reverse order
    pad_sizes.reverse()

    # Pad the tensor if needed
    if any(pad_sizes):
        tensor = F.pad(tensor, pad_sizes)

    # Crop the tensor if needed
    if any(s.start is not None or s.stop is not None for s in crop_slices):
        tensor = tensor[tuple(crop_slices)]

    return tensor


def crop_tensor_to_shape(tensor: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """
    Crops the tensor to the specified shape, regardless of the number of dimensions.

    Args:
        tensor (torch.Tensor): The tensor to be cropped.
        shape (List[int]): The desired shape to crop to. It should have the same length as the number of dimensions in the tensor.

    Returns:
        torch.Tensor: The cropped tensor with the specified shape.
    """
    if len(tensor.shape) != len(shape):
        raise ValueError(
            "The shape list must have the same number of dimensions as the tensor.")

    # Dynamically create the slicing for each dimension
    slices = tuple(slice(0, dim) for dim in shape)

    # Apply the slices to crop the tensor
    tensor_cropped = tensor[slices]

    return tensor_cropped


def get_new_voxelmorph_image_shape(old_shape: List[int],
                                   number_of_layers_in_encoder: int) -> List[int]:
    """
    Input shape for voxelmorph must be multiples of 2^n,
    for N being the number of layers in the encoder.
    We only pad, to not loose any information.
    """

    new_shape: List[int] = [0, 0, 0]

    for i in range(3):
        new_shape[i] = torch.ceil(torch.tensor(
            old_shape[i] / (2 ** number_of_layers_in_encoder))).int().item() * (2 ** number_of_layers_in_encoder)

    return new_shape
