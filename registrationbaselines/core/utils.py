import pandas as pd
from pathlib import Path

from typing import Any, List, Tuple

from scipy.ndimage import map_coordinates
from sympy import true
import yaml
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from registrationbaselines.core import utils_metrics

from registrationbaselines.core.types import floatArray2D, floatArray3Dor4D, floarArray4Dor5D, array2Dor3D, floatArray2Dor3D


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

    grid = utils_metrics.compute_grid(
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


def pad_tensor_to_shape(tensor: torch.Tensor,
                        new_shape: List[int]) -> torch.Tensor:
    """
    Input shape for voxelmorph must be multiples of 2^n,
    for N being the number of layers in the encoder.
    We only pad, to not loose any information.
    """

    shape = tensor.shape

    if shape[0:2] != [1, 1] and len(shape) != 5:
        raise ValueError(
            "The input tensor should be of shape (1,1,H,W,D).")

    shape = shape[2:]
    pad: List[int] = [0, 0, 0]

    for i in range(3):
        size = shape[i]
        pad[i] = new_shape[i] - size  # Only pad at the end

    # Reverse pad list and interleave with zeros for F.pad format
    pad = [item for sublist in zip([0]*3, reversed(pad))
           for item in sublist]

    if pad == [0, 0, 0, 0, 0, 0]:
        return tensor

    padded = F.pad(tensor, pad)

    return padded


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
    assert len(new_shape) == tensor.dim(), "New shape must have the same number of dimensions as the input tensor"
    
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
