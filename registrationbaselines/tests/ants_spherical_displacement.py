import sys
from pathlib import Path

from typing import Tuple, Any

import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
import torch
import math
import torch.nn.functional as F
import ants

import registrationbaselines.displacement.utils_displacement

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.core import utils  # nopep8
from registrationbaselines.evaluation.plot_objects import plot_all_registration_results  # nopep8


def create_concentric_ellipsoids(shape: Tuple[int, int, int] = (200, 200, 200),
                                 semi_axes: Tuple[float, float, float] = (
                                     100.0, 80.0, 60.0),
                                 ellipsoid_thickness: float = 5.0,
                                 ellipsoid_spacing: float = 10.0,
                                 num_ellipsoids: int = 5) -> torch.Tensor:

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

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


def create_displacement_field_sphere(
        shape: Tuple[int, int, int],
        semi_axes: Tuple[float, float, float],
        max_displacement: float,
        direction: Tuple[int, int, int]
) -> torch.Tensor:

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Move to center and calculate normalized ellipsoidal radius
    center = torch.tensor([(s - 1) / 2 for s in shape])
    x = (x - center[2]) / semi_axes[0]
    y = (y - center[1]) / semi_axes[1]
    z = (z - center[0]) / semi_axes[2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # Calculate displacement magnitude
    disp_mag = torch.zeros_like(r)
    mask = r <= 1.0  # within ellipsoid
    disp_mag[mask] = max_displacement * torch.sin(math.pi * r[mask])

    # Calculate unit vectors (use original x, y, z for directions)
    eps = 1e-8  # to avoid division by zero
    unit_x = x / (r + eps) * 1
    unit_x = x / (r + eps) * 1
    unit_y = y / (r + eps) * 1
    unit_z = z / (r + eps) * 1

    # Create displacement field
    displacement = torch.stack([
        disp_mag * unit_x * direction[0],
        disp_mag * unit_y * direction[1],
        disp_mag * unit_z * direction[2],
    ], dim=-1)

    return displacement


shape = (201, 201, 201)
thickness = 1
sphere_spacing = 20
num_ellipsoids = 3
semi_axes_spheres = (50, 50, 50)
semi_axes_displacement = (70, 70, 70)
max_displacement = -20

image = create_concentric_ellipsoids(shape,
                                     semi_axes_spheres,
                                     thickness,
                                     sphere_spacing,
                                     num_ellipsoids + 1)
displacement_x = create_displacement_field_sphere(shape,
                                                  semi_axes_displacement,
                                                  max_displacement,
                                                  (1, 0, 0))
displacement_y = create_displacement_field_sphere(shape,
                                                  semi_axes_displacement,
                                                  max_displacement,
                                                  (0, 1, 0))
displacement_z = create_displacement_field_sphere(shape,
                                                  semi_axes_displacement,
                                                  max_displacement,
                                                  (0, 0, 1))
displacement_xyz = create_displacement_field_sphere(shape,
                                                    semi_axes_displacement,
                                                    max_displacement,
                                                    (1, 1, 1))


# deformed_image_torch_x = utils.deform_image(image,
#                                             displacement_x.detach().clone())
# deformed_image_torch_y = utils.deform_image(image,
#                                             displacement_y.detach().clone())
# deformed_image_torch_z = utils.deform_image(image,
#                                             displacement_z.detach().clone())
# deformed_image_torch_xyz = utils.deform_image(image,
#                                               displacement_xyz.detach().clone())

deformed_image_ants_x = utils.deform_image_ants(image,
                                                displacement_x.detach().clone())  # [..., [2, 1, 0]])
deformed_image_ants_y = utils.deform_image_ants(image,
                                                displacement_y.detach().clone())  # [..., [2, 1, 0]])
deformed_image_ants_z = utils.deform_image_ants(image,
                                                displacement_z.detach().clone())  # [..., [2, 1, 0]])
deformed_image_ants_xyz = utils.deform_image_ants(image,
                                                  displacement_xyz.detach().clone())  # [..., [2, 1, 0]])

displacement_unit_x = registrationbaselines.displacement.utils_displacement.displacement_to_unit_displacement(
    displacement_x.detach().clone())
displacement_unit_y = registrationbaselines.displacement.utils_displacement.displacement_to_unit_displacement(
    displacement_y.detach().clone())
displacement_unit_z = registrationbaselines.displacement.utils_displacement.displacement_to_unit_displacement(
    displacement_z.detach().clone())
displacement_unit_xyz = registrationbaselines.displacement.utils_displacement.displacement_to_unit_displacement(
    displacement_xyz.detach().clone())

# plot_all_registration_results(moving_image=image,
#                               fixed_image=image,
#                               pred_image=deformed_image_torch_x,
#                               pred_segmentations=None,
#                               fixed_segmentations=None,
#                               moving_keypoints=None,
#                               pred_keypoints=None,
#                               fixed_keypoints=None,
#                               displacement=displacement_unit_x.detach().clone(),
#                               save_path="images/displacement_spherical_x_torch.png")
# plot_all_registration_results(moving_image=image,
#                               fixed_image=image,
#                               pred_image=deformed_image_torch_y,
#                               pred_segmentations=None,
#                               fixed_segmentations=None,
#                               moving_keypoints=None,
#                               pred_keypoints=None,
#                               fixed_keypoints=None,
#                               displacement=displacement_unit_y.detach().clone(),
#                               save_path="images/displacement_spherical_y_torch.png")
# plot_all_registration_results(moving_image=image,
#                               fixed_image=image,
#                               pred_image=deformed_image_torch_z,
#                               pred_segmentations=None,
#                               fixed_segmentations=None,
#                               moving_keypoints=None,
#                               pred_keypoints=None,
#                               fixed_keypoints=None,
#                               displacement=displacement_unit_z.detach().clone(),
#                               save_path="images/displacement_spherical_z_torch.png")
# plot_all_registration_results(moving_image=image,
#                               fixed_image=image,
#                               pred_image=deformed_image_torch_xyz,
#                               pred_segmentations=None,
#                               fixed_segmentations=None,
#                               moving_keypoints=None,
#                               pred_keypoints=None,
#                               fixed_keypoints=None,
#                               displacement=displacement_unit_xyz.detach().clone(),
#                               save_path="images/displacement_spherical_xyz_torch.png")

plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_ants_x,
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_x.detach().clone(),
                              save_path="images/displacement_spherical_x_ants.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_ants_y,
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_y.detach().clone(),
                              save_path="images/displacement_spherical_y_ants.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_ants_z,
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_z.detach().clone(),
                              save_path="images/displacement_spherical_z_ants.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_ants_xyz,
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_xyz.detach().clone(),
                              save_path="images/displacement_spherical_xyz_ants.png")
