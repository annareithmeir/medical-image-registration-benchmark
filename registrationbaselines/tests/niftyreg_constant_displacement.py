import sys
from pathlib import Path

from typing import Tuple

import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib

import registrationbaselines.warping.deform_objects
import registrationbaselines.warping.utils_displacement

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.core import utils_nifti  # nopep8
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


def create_displacement_constant(
        shape: Tuple[int, int, int],
        displacement_magnitude: float,
        direction: Tuple[int, int, int]
) -> torch.Tensor:

    # Create coordinate grid
    z, y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(
        shape[1]), torch.arange(shape[2]), indexing='ij')

    # Normalize x coordinate
    x = x / (shape[2] - 1)

    # Calculate displacement magnitude
    disp_mag = torch.ones_like(x) * displacement_magnitude

    # Create unit vectors for displacement (only in y-direction)
    unit_x = torch.ones_like(disp_mag)
    unit_y = torch.ones_like(disp_mag)
    unit_z = torch.ones_like(disp_mag)

    # Create displacement field
    displacement = torch.stack([
        disp_mag * unit_z * direction[0],
        disp_mag * unit_x * direction[1],
        disp_mag * unit_y * direction[2]
    ], dim=-1)

    return displacement


shape = (101, 151, 201)
thickness = 4
sphere_spacing = 20
num_ellipsoids = 3
semi_axes = (55, 45, 25)
displacement_size = np.asarray([90, 90, 90])
max_displacement = 11

image = create_concentric_ellipsoids(shape,
                                     semi_axes,
                                     thickness,
                                     sphere_spacing,
                                     num_ellipsoids + 1)
displacement_x = create_displacement_constant(shape,
                                              max_displacement,
                                              (1, 0, 0))
displacement_y = create_displacement_constant(shape,
                                              max_displacement,
                                              (0, 1, 0))
displacement_z = create_displacement_constant(shape,
                                              max_displacement,
                                              (0, 0, 1))
displacement_xyz = create_displacement_constant(shape,
                                                max_displacement,
                                                (1, 1, 1))

displacement_x_save = displacement_x  # .permute(3, 0, 1, 2)
displacement_sitk = sitk.GetImageFromArray(
    displacement_x_save.cpu().numpy())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
displacement_x = displacement_x.to(device)
deformed_image_torch_x = registrationbaselines.warping.deform_objects.deform_image(image,
                                                                                   displacement_x.detach().clone())

registrationbaselines.warping.utils_displacement.print_histogram(
    deformed_image_torch_x, 10)

im_moving = sitk.GetImageFromArray(image.cpu().numpy())
im_fixed = sitk.GetImageFromArray(deformed_image_torch_x.cpu().numpy())

sitk.WriteImage(displacement_sitk,
                "registrationbaselines/tests/images/displacement_x_11.nii.gz")
sitk.WriteImage(
    im_moving, "registrationbaselines/tests/images/moving_x_11.nii.gz")
sitk.WriteImage(
    im_fixed, "registrationbaselines/tests/images/fixed_x_11.nii.gz")

"""
deformed_image_torch_y = utils.deform_image(image,
                                            displacement_y.detach().clone())
deformed_image_torch_z = utils.deform_image(image,
                                            displacement_z.detach().clone())
deformed_image_torch_xyz = utils.deform_image(image,
                                              displacement_xyz.detach().clone())
"""

"""
deformed_image_niftyreg_y = utils.deform_image_niftyreg(image,
                                                    displacement_y.detach().clone())
deformed_image_niftyreg_z = utils.deform_image_niftyreg(image,
                                                    displacement_z.detach().clone())
deformed_image_niftyreg_xyz = utils.deform_image_niftyreg(image,
                                                      displacement_xyz.detach().clone())
"""

displacement_unit_x = registrationbaselines.warping.utils_displacement.displacement_to_unit_displacement(
    displacement_x.detach().clone())
"""
displacement_unit_y = utils.displacement_to_unit_displacement(
    displacement_y.detach().clone())
displacement_unit_z = utils.displacement_to_unit_displacement(
    displacement_z.detach().clone())
displacement_unit_xyz = utils.displacement_to_unit_displacement(
    displacement_xyz.detach().clone())
"""

image = image.cpu()
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_torch_x.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_x.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_x_torch.png")
"""
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_torch_y.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_y.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_y_torch.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_torch_z.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_z.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_z_torch.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_torch_xyz.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_z.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_xyz_torch.png")


plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_niftyreg_x.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_x.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_x_niftyreg.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_niftyreg_y.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_y.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_y_niftyreg.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_niftyreg_z.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_z.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_z_niftyreg.png")
plot_all_registration_results(moving_image=image,
                              fixed_image=image,
                              pred_image=deformed_image_niftyreg_xyz.cpu(),
                              pred_segmentations=None,
                              fixed_segmentations=None,
                              moving_keypoints=None,
                              pred_keypoints=None,
                              fixed_keypoints=None,
                              displacement=displacement_unit_z.detach().clone().cpu(),
                              save_path="registrationbaselines/tests/images/displacement_constant_xyz_niftyreg.png")
"""
