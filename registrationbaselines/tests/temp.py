from pathlib import Path
import sys

from typing import Tuple, Any

import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
import torch
import math
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

# import registrationbaselines.core.utils_nifti as utils_nifti
from registrationbaselines.core.visualization import plot_all_registration_results
import registrationbaselines.core.utils as utils


def create_displacement_field_sphere(
        shape: Tuple[int, int, int],
        semi_axes: np.ndarray[int, np.dtype[np.float32]],
        max_displacement: float = 10.0
) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Move to center and calculate normalized ellipsoidal radius
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    x = (x.to(device) - center[2]) / semi_axes[0]
    y = (y.to(device) - center[1]) / semi_axes[1]
    z = (z.to(device) - center[0]) / semi_axes[2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # Calculate displacement magnitude
    disp_mag = torch.zeros_like(r)
    mask = r <= 1.0  # within ellipsoid
    disp_mag[mask] = max_displacement * torch.sin(math.pi * r[mask])

    # Calculate unit vectors (use original x, y, z for directions)
    eps = 1e-8  # to avoid division by zero
    unit_x = x / (r + eps) * 1
    unit_y = y / (r + eps) * 1
    unit_z = z / (r + eps) * 1

    # Create displacement field
    displacement = torch.stack([
        disp_mag * unit_x,
        disp_mag * unit_y,
        disp_mag * unit_z
    ], dim=-1)

    return displacement


def create_displacement_field_one(
    shape: tuple[int, int, int],
    direction: str,
    max_displacement: float,
    spatial_extent: np.ndarray[int, np.dtype[np.float32]]
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Move to center and calculate normalized ellipsoidal radius
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    x = (x.to(device) - center[2]) / semi_axes[0]
    y = (y.to(device) - center[1]) / semi_axes[1]
    z = (z.to(device) - center[0]) / semi_axes[2]

    # Select the direction for displacement
    if direction == 'x':
        r = torch.abs(x)
    elif direction == 'y':
        r = torch.abs(y)
    elif direction == 'z':
        r = torch.abs(z)
    else:
        raise ValueError("Direction must be one of 'x', 'y', or 'z'.")

    # Calculate displacement magnitude
    disp_mag = torch.zeros_like(r)
    mask = r <= 1.0  # within extent
    disp_mag[mask] = max_displacement * torch.sin(math.pi * r[mask])

    # Create displacement field based on the selected direction
    if direction == 'x':
        displacement = torch.stack([disp_mag, torch.zeros_like(
            disp_mag), torch.zeros_like(disp_mag)], dim=-1)
    elif direction == 'y':
        displacement = torch.stack(
            [torch.zeros_like(disp_mag), disp_mag, torch.zeros_like(disp_mag)], dim=-1)
    elif direction == 'z':
        displacement = torch.stack(
            [torch.zeros_like(disp_mag), torch.zeros_like(disp_mag), disp_mag], dim=-1)

    return displacement


def create_displacement_field(
    shape: tuple[int, int, int],
    direction: str,
    max_displacement: float,
    spatial_extent: np.ndarray[int, np.dtype[np.float32]]
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    displacement = torch.zeros(shape)

    max = 20
    min = 0

    val_range = np.linspace(min, max, shape[0])

    for i in range(shape[0]):
        displacement[i, ...] = val_range[i]

    displacement = displacement.to(device)

    return torch.stack(
        [displacement, torch.zeros_like(displacement), torch.zeros_like(displacement)], dim=-1)


def create_concentric_spheres(shape: Tuple[int, int, int] = (200, 200, 200),
                              sphere_thickness: float = 5.0,
                              sphere_spacing: float = 10.0,
                              num_spheres: int = 5) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Move to center and calculate radius
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    x = x.to(device) - center[2]
    y = y.to(device) - center[1]
    z = z.to(device) - center[0]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # Create volume
    volume = torch.zeros(shape, device=device)

    for i in range(1, num_spheres):
        if i == 1:
            spacing = sphere_spacing / 2
        else:
            spacing = sphere_spacing
        inner_radius = i * (sphere_thickness + spacing)
        outer_radius = inner_radius + sphere_thickness

        # Create sphere
        sphere_mask = (r >= inner_radius) & (r < outer_radius)
        volume[sphere_mask] = 1.0

    return volume


def create_concentric_ellipsoids(shape: Tuple[int, int, int] = (200, 200, 200),
                                 semi_axes: Tuple[float, float, float] = (
                                     100.0, 80.0, 60.0),
                                 ellipsoid_thickness: float = 5.0,
                                 ellipsoid_spacing: float = 10.0,
                                 num_ellipsoids: int = 5) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Move to center and calculate normalized ellipsoidal radius
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    x = (x.to(device) - center[2]) / semi_axes[0]
    y = (y.to(device) - center[1]) / semi_axes[1]
    z = (z.to(device) - center[0]) / semi_axes[2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # Create volume
    volume = torch.zeros(shape, device=device)

    for i in range(1, num_ellipsoids):
        if i == 1:
            spacing = ellipsoid_spacing / 2
        else:
            spacing = ellipsoid_spacing

        inner_radius = i * (ellipsoid_thickness + spacing) / max(semi_axes)
        outer_radius = inner_radius + ellipsoid_thickness / max(semi_axes)

        # Create ellipsoid
        ellipsoid_mask = (r >= inner_radius) & (r < outer_radius)
        volume[ellipsoid_mask] = 1.0

    return volume


def create_sphere_keypoints(shape: Tuple[int, int, int] = (200, 200, 200),
                            sphere_thickness: float = 5.0,
                            sphere_spacing: float = 10.0,
                            num_spheres: int = 5) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    keypoints = []

    for i in range(1, num_spheres):

        if i == 1:
            spacing = sphere_spacing / 2
        else:
            spacing = sphere_spacing

        radius = i * (sphere_thickness + spacing) + sphere_thickness / 2

        # Create 6 keypoints for each sphere
        sphere_keypoints = [
            # x-y plane (2 points)
            [center[0] + radius, center[1], center[2]],
            [center[0] - radius, center[1], center[2]],

            # y-z plane (2 points)
            [center[0], center[1] + radius, center[2]],
            [center[0], center[1] - radius, center[2]],

            # x-z plane (2 points)
            [center[0], center[1], center[2] + radius],
            [center[0], center[1], center[2] - radius]
        ]

        keypoints.extend(sphere_keypoints)

    return torch.tensor(keypoints, device=device)


def create_ellipsoid_keypoints(shape: Tuple[int, int, int] = (200, 200, 200),
                               semi_axes: Tuple[float, float, float] = (
                                   100.0, 80.0, 60.0),
                               ellipsoid_thickness: float = 5.0,
                               ellipsoid_spacing: float = 10.0,
                               num_ellipsoids: int = 5) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    keypoints = []

    for i in range(1, num_ellipsoids):
        if i == 1:
            spacing = ellipsoid_spacing / 2
        else:
            spacing = ellipsoid_spacing

        radius = i * (ellipsoid_thickness + spacing) + ellipsoid_thickness / 2

        # Create keypoints for each ellipsoid based on semi_axes
        ellipsoid_keypoints = [
            # x-axis keypoints (along x-axis, keep y and z at center)
            [center[0], center[1], center[2] + \
                radius * semi_axes[0] / max(semi_axes)],
            [center[0], center[1], center[2] - \
                radius * semi_axes[0] / max(semi_axes)],

            # y-axis keypoints (along y-axis, keep x and z at center)
            [center[0], center[1] + radius * \
                semi_axes[1] / max(semi_axes), center[2]],
            [center[0], center[1] - radius * \
                semi_axes[1] / max(semi_axes), center[2]],

            # z-axis keypoints (along z-axis, keep x and y at center)
            [center[0] + radius * semi_axes[2] / \
                max(semi_axes), center[1], center[2]],
            [center[0] - radius * semi_axes[2] / \
                max(semi_axes), center[1], center[2]],
        ]

        keypoints.extend(ellipsoid_keypoints)

    return torch.tensor(keypoints, device=device)


def create_cuboid_keypoints(
    shape: tuple[int, int, int],
    cuboid_dimensions: np.ndarray[int, np.dtype[np.float32]],
    cuboid_thickness: float = 2.0,
    cuboid_spacing: float = 4.0,
    num_cuboids: int = 3
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    keypoints = []

    for i in range(num_cuboids):
        # Calculate the size of the current cuboid
        current_size = [
            dim + i * (cuboid_thickness + cuboid_spacing) * 2 for dim in cuboid_dimensions]

        # Calculate the half-size of the current cuboid (to the center of the thickness)
        half_size = [(size + cuboid_thickness) / 2 for size in current_size]

        # Create keypoints for each cuboid
        cuboid_keypoints = [
            # x-axis keypoints (along x-axis, keep y and z at center)
            [center[0], center[1], center[2] + half_size[2]],
            [center[0], center[1], center[2] - half_size[2]],
            # y-axis keypoints (along y-axis, keep x and z at center)
            [center[0], center[1] + half_size[1], center[2]],
            [center[0], center[1] - half_size[1], center[2]],
            # z-axis keypoints (along z-axis, keep x and y at center)
            [center[0] + half_size[0], center[1], center[2]],
            [center[0] - half_size[0], center[1], center[2]],
        ]
        keypoints.extend(cuboid_keypoints)

    return torch.tensor(keypoints, device=device)


def create_concentric_cuboids(shape: Tuple[int, int, int],
                              cuboid_dimensions: np.ndarray[int, np.dtype[np.float32]],
                              cuboid_thickness: float = 5.0,
                              cuboid_spacing: float = 10.0,
                              num_cuboids: int = 5) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Move to center
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    x = x.to(device) - center[2]
    y = y.to(device) - center[1]
    z = z.to(device) - center[0]

    # Create volume
    volume = torch.zeros(shape, device=device)

    for i in range(num_cuboids):
        inner_size = [
            dim + i * (cuboid_thickness + cuboid_spacing) * 2 for dim in cuboid_dimensions]
        outer_size = [inner + cuboid_thickness * 2 for inner in inner_size]

        # Create cuboid mask
        cuboid_mask = (
            (x.abs() <= outer_size[0] / 2) &
            (y.abs() <= outer_size[1] / 2) &
            (z.abs() <= outer_size[2] / 2)
        ) & ~(
            (x.abs() < inner_size[0] / 2) &
            (y.abs() < inner_size[1] / 2) &
            (z.abs() < inner_size[2] / 2)
        )

        volume[cuboid_mask] = 1.0

    return volume


def print_square_tensor(tensor: torch.Tensor) -> None:
    """
    Nicely prints a 2D square torch tensor with 2 values after the decimal point.
    Positive values are aligned with negative values by adding a space in front.
    """
    if tensor.dim() != 2 or tensor.size(0) != tensor.size(1):
        raise ValueError("Input must be a 2D square tensor.")

    with torch.no_grad():
        for row in tensor:
            print(" ".join(f"{item:5.2f}" for item in row))


torch.set_printoptions(linewidth=200)

shape = (201, 201, 201)
thickness = 1
sphere_spacing = 30
num_ellipsoids = 3
semi_axes = np.asarray([80, 80, 80])
displacement_size = np.asarray([90, 90, 90])
max_displacement = 20

image = create_concentric_cuboids(shape,
                                  semi_axes/2,
                                  thickness,
                                  sphere_spacing,
                                  num_ellipsoids + 1)
# image = create_concentric_ellipsoids(shape,
#                                      semi_axes,
#                                      thickness,
#                                      sphere_spacing,
#                                      num_ellipsoids + 1)

keypoints = create_cuboid_keypoints(shape,
                                    semi_axes/2,
                                    thickness,
                                    sphere_spacing,
                                    num_ellipsoids)
# keypoints = create_ellipsoid_keypoints(shape,
#                                        semi_axes,
#                                        thickness,
#                                        sphere_spacing,
#                                        num_ellipsoids + 1)

displacement = create_displacement_field(shape,
                                         'x',
                                         max_displacement,
                                         displacement_size)
# displacement = create_displacement_field_sphere(shape,
#                                          displacement_size,
#                                          -max_displacement)

displacement_unit = utils.displacement_to_unit_displacement(
    displacement.detach().clone())

deformed_image = utils.deform_image(image,
                                    displacement_unit.detach().clone())
deformed_keypoints = utils.deform_keypoints(keypoints,
                                            displacement_unit.detach().clone())

# displacement_unit = displacement_unit.permute(2, 1, 0, 3)

plot_all_registration_results(save_path=Path("registrationbaselines/tests/test_files/temp_vis.png"),
                              moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image,
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=keypoints,
                              pred_keypoints=deformed_keypoints,
                              fixed_keypoints=keypoints,
                              displacement=displacement_unit.detach().clone())

x = 0

disp_x = displacement[..., 0]
disp_y = displacement[..., 1]
disp_z = displacement[..., 2]

utils.save_image(image,
                 Path("registrationbaselines/tests/test_files/image_tmp.nii.gz"),
                 (1, 1, 1))
utils.save_image(deformed_image,
                 Path("registrationbaselines/tests/test_files/image_deformed_tmp.nii.gz"),
                 (1, 1, 1))
utils.save_displacement(displacement,
                        Path(
                            "registrationbaselines/tests/test_files/displacement_tmp.nii.gz"),
                        (1, 1, 1, 1))
utils.save_image(disp_x,
                 Path("registrationbaselines/tests/test_files/displacement_x_tmp.nii.gz"),
                 (1, 1, 1))
utils.save_image(disp_y,
                 Path("registrationbaselines/tests/test_files/displacement_y_tmp.nii.gz"),
                 (1, 1, 1))
utils.save_image(disp_z,
                 Path("registrationbaselines/tests/test_files/displacement_z_tmp.nii.gz"),
                 (1, 1, 1))
