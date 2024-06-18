from pathlib import Path
import sys
import pickle
import copy

import numpy as np
import torch
from torch.nn.parameter import Parameter
import SimpleITK as sitk
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8


import latent_space_registration.airlab as al
import latent_space_registration.airlab.transformation as al_transformation
import latent_space_registration.airlab.loss as al_loss
import latent_space_registration.airlab.regulariser as al_regulariser


def apply_displacement_field(image_fixed: sitk.Image,
                             image_moving: sitk.Image,
                             displacement: sitk.Image,
                             sitk_interpolator: int):
    """
    Apply a deformation to an image using the provided deformation.
    """
    # Create the transform using the displacement field
    displacement_field_transform = sitk.DisplacementFieldTransform(
        displacement)

    # Apply the transform to the input image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_fixed)
    resampler.SetInterpolator(sitk_interpolator)
    resampler.SetTransform(displacement_field_transform)

    deformed_image = resampler.Execute(image_moving)

    return deformed_image


def create_grid_image(size, grid_size, line_thickness):
    """
    Creates a white square image with a black grid using numpy.

    :param size: The size of the image (width and height).
    :param grid_size: The size of the grid squares.
    :param line_thickness: The thickness of the grid lines.
    :return: A numpy array representing the image.
    """
    # Create a white square image (3 channels for RGB)
    img = np.ones((size[0], size[1]), dtype=np.uint8) * 255

    # Draw horizontal lines
    for y in range(0, size[0], grid_size):
        img[y:y+line_thickness, :] = 0  # Set the rows to black

    # Draw vertical lines
    for x in range(0, size[1], grid_size):
        img[:, x:x+line_thickness] = 0  # Set the columns to black

    return img


def warp_landmarks(keypoints, displacement, image_size, device):
    # Create a grid for the landmarks
    keypoints_tensor = torch.tensor(
        keypoints, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Normalize keypoints to range [-1, 1] for grid_sample
    keypoints_normalized = keypoints_tensor.clone()
    keypoints_normalized[..., 0] = 2.0 * \
        keypoints_tensor[..., 0] / (image_size[0] - 1) - 1.0
    keypoints_normalized[..., 1] = 2.0 * \
        keypoints_tensor[..., 1] / (image_size[1] - 1) - 1.0

    # Apply displacement
    displacement_grid = F.grid_sample(
        displacement.unsqueeze(0), keypoints_normalized, align_corners=True)

    # Denormalize back to original image coordinates
    deformed_landmarks = displacement_grid.squeeze().detach().cpu().numpy()

    deformed_landmarks[..., 0] = (
        deformed_landmarks[..., 0] + 1.0) * (image_size[0] - 1) / 2.0
    deformed_landmarks[..., 1] = (
        deformed_landmarks[..., 1] + 1.0) * (image_size[1] - 1) / 2.0

    return deformed_landmarks


image_moving_shape = (2912, 2100)
grid_spacing = 200
line_thickness = 5
dtype = torch.float32
device = torch.device("cuda:0")
sigma = [500, 500]


# Example usage
image_fixed = create_grid_image(
    image_moving_shape, grid_spacing, line_thickness)
image_moving = create_grid_image(
    image_moving_shape, grid_spacing, line_thickness)

image_moving = al.utils.image_from_numpy(
    image_moving, [1, 1], [0, 0], dtype=dtype, device=device)

transformation = al_transformation.pairwise.BsplineTransformation(image_moving.size,
                                                                  sigma=sigma,
                                                                  rgb=False,
                                                                  order=1,
                                                                  dtype=dtype,
                                                                  device=device,
                                                                  diffeomorphic=True)

tx = 0.3
ty = 0.1

p = transformation.trans_parameters.detach()
p[0, 0, 5, 4] = tx
p[0, 1, 5, 4] = ty
transformation.trans_parameters = Parameter(p)

displacement = transformation.get_displacement()


# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
"""
displacement_save = al_transformation.utils.unit_displacement_to_displacement(
    copy.copy(displacement))
displacement_save = al.create_displacement_image_from_image(
    displacement_save, copy.copy(image_moving))
sitk.WriteImage(displacement_save.itk(),
                '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/displacement.nii.gz')
image_moving_save = copy.copy(image_moving).itk()
sitk.WriteImage(image_moving_save,
                '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/image_moving.nii.gz')


displacement_load = sitk.ReadImage(
    '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/displacement.nii.gz')
image_moving_load = sitk.ReadImage(
    '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/image_moving.nii.gz')

warped_image_load = apply_displacement_field(image_moving_load,
                                             image_moving_load,
                                             sitk.Cast(
                                                 displacement_load, sitk.sitkVectorFloat64),
                                             sitk.sitkLinear)

warped_image_load = al.create_tensor_image_from_itk_image(
    warped_image_load, device="cuda:0")
"""
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
warped_image = al_transformation.utils.warp_image(
    image_moving, displacement)

new_image = warped_image.image.detach().cpu().numpy().squeeze()


# p = transformation.trans_parameters.detach()
# p[0, 0, 5, 4] = -tx
# p[0, 1, 5, 4] = -ty
# transformation.trans_parameters = Parameter(p)

# displacement_points = transformation.get_displacement()

keypoints = np.array([[2200.0, 1600.0], [80.0, 10.0], [7, 49]])

displacement = al_transformation.utils.unit_displacement_to_displacement(
    displacement)  # unit measures to image domain measures
displacement = al.create_displacement_image_from_image(
    displacement, image_moving)

deformed_landmarks = al.utils.points.Points.transform(
    keypoints, displacement)

# deformed_landmarks = warp_landmarks(
#     keypoints, displacement, image_moving.size, device)

plt.close()
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot warped_image
axs[0].imshow(new_image)
axs[0].scatter(keypoints[:, 1], keypoints[:, 0], c='k',
               marker='*', label='Original Keypoints')
axs[0].scatter(deformed_landmarks[:, 1], deformed_landmarks[:, 0],
               c='r', marker='x', label='Deformed Landmarks')
axs[0].legend()
axs[0].set_title('Warped Image')

# Plot warped_image_load
# warped_image_load_np = warped_image_load.image.detach().cpu().numpy().squeeze()
# axs[1].imshow(warped_image_load_np)
# # axs[1].scatter(keypoints[:, 0], keypoints[:, 1], c='b',
# #                marker='x', label='Original Keypoints')
# # axs[1].scatter(deformed_landmarks[:, 0], deformed_landmarks[:, 1],
# #                c='g', marker='x', label='Deformed Landmarks')
# axs[1].legend()
# axs[1].set_title('Warped Image Load')

# # Plot difference image
# difference_image = np.abs(new_image - warped_image_load_np)
# axs[2].imshow(difference_image)
# axs[2].set_title(f'Difference Image: {np.sum(difference_image)}')

plt.show()
plt.savefig(
    "/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/grid.png")
plt.close()


x = 0
