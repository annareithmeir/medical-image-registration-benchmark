from pathlib import Path
import sys

from typing import Tuple

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


def create_displacement_field(
        shape: Tuple[int, int, int] = (200, 200, 200),
        semi_axes: Tuple[float, float, float] = (100.0, 80.0, 60.0),
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
    unit_x = x / (r + eps) * 0
    unit_y = y / (r + eps) * 1
    unit_z = z / (r + eps) * 0

    # Create displacement field
    displacement = torch.stack([
        disp_mag * unit_x,
        disp_mag * unit_y,
        disp_mag * unit_z
    ], dim=-1)

    return displacement


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


def create_cube_keypoints(shape: Tuple[int, int, int] = (200, 200, 200),
                          side_lengths: Tuple[float, float, float] = (
                              100.0, 80.0, 60.0),
                          cube_thickness: float = 5.0,
                          cube_spacing: float = 10.0,
                          num_cubes: int = 5) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)
    keypoints = []

    for i in range(1, num_cubes):
        if i == 1:
            spacing = cube_spacing / 2
        else:
            spacing = cube_spacing

        offset = i * (cube_thickness + spacing) + cube_thickness / 2

        # Create keypoints for each cube based on side_lengths
        cube_keypoints = [
            # x-axis keypoints (along x-axis, keep y and z at center)
            [center[0], center[1], center[2] + offset],
            [center[0], center[1], center[2] - offset],

            # y-axis keypoints (along y-axis, keep x and z at center)
            [center[0], center[1] + offset, center[2]],
            [center[0], center[1] - offset, center[2]],

            # z-axis keypoints (along z-axis, keep x and y at center)
            [center[0] + offset, center[1], center[2]],
            [center[0] - offset, center[1], center[2]],
        ]

        keypoints.extend(cube_keypoints)

    return torch.tensor(keypoints, device=device)


def create_concentric_cuboids(
    shape: Tuple[int, int, int] = (200, 200, 200),
    cuboid_size: Tuple[float, float, float] = (100.0, 80.0, 60.0),
    cuboid_thickness: float = 5.0,
    cuboid_spacing: float = 10.0,
    num_cuboids: int = 5
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create volume
    volume = torch.zeros(shape, device=device)

    # Calculate center of the volume
    center = torch.tensor([(s - 1) / 2 for s in shape], device=device)

    for i in range(num_cuboids):
        # Calculate the current cuboid dimensions
        current_size = [
            cuboid_size[j] + 2 * i * (cuboid_thickness + cuboid_spacing)
            for j in range(3)
        ]

        # Calculate the start and end indices for each dimension
        starts = [int(center[j] - current_size[j] / 2) for j in range(3)]
        ends = [int(center[j] + current_size[j] / 2) for j in range(3)]

        # Check if the current cuboid fits within the volume
        if any(starts[j] < 0 or ends[j] > shape[j] for j in range(3)):
            print(
                f"Warning: Cuboid {i+1} doesn't fit in the volume. Skipping.")
            break

        # Create the outer surface of the cuboid
        volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] = 1.0

        # Remove the inner part to create a hollow cuboid (except for the innermost cuboid)
        if i < num_cuboids - 1:
            inner_starts = [starts[j] + cuboid_thickness for j in range(3)]
            inner_ends = [ends[j] - cuboid_thickness for j in range(3)]
            volume[inner_starts[0]:inner_ends[0],
                   inner_starts[1]:inner_ends[1],
                   inner_starts[2]:inner_ends[2]] = 0.0

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


shape = (201, 201, 201)
thickness = 1
sphere_spacing = 10
num_ellipsoids = 3
semi_axes = (100, 80, 60)
max_displacement = 2

image = create_concentric_cuboids(shape,
                                  semi_axes,
                                  thickness,
                                  sphere_spacing,
                                  num_ellipsoids + 1)
# image = create_concentric_ellipsoids(shape,
#                                      semi_axes,
#                                      thickness,
#                                      sphere_spacing,
#                                      num_ellipsoids + 1)

keypoints = create_cube_keypoints(shape,
                                  semi_axes,
                                  thickness,
                                  sphere_spacing,
                                  num_ellipsoids + 1)
# keypoints = create_ellipsoid_keypoints(shape,
#                                        semi_axes,
#                                        thickness,
#                                        sphere_spacing,
#                                        num_ellipsoids + 1)

displacement = create_displacement_field(shape,
                                         semi_axes,
                                         -max_displacement)

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
