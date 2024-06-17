from pathlib import Path
import sys
import pickle

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


def create_grid_image(size, grid_size, line_thickness):
    """
    Creates a white square image with a black grid using numpy.

    :param size: The size of the image (width and height).
    :param grid_size: The size of the grid squares.
    :param line_thickness: The thickness of the grid lines.
    :return: A numpy array representing the image.
    """
    # Create a white square image (3 channels for RGB)
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Draw horizontal lines
    for y in range(0, size, grid_size):
        img[y:y+line_thickness, :] = 0  # Set the rows to black

    # Draw vertical lines
    for x in range(0, size, grid_size):
        img[:, x:x+line_thickness] = 0  # Set the columns to black

    return img


image_moving_shape = (2912, 2912)
grid_spacing = 200
line_thickness = 5
dtype = torch.float32
device = torch.device("cuda:0")
sigma = [500, 500]


# Example usage
image_fixed = create_grid_image(
    image_moving_shape[0], grid_spacing, line_thickness)
image_moving = create_grid_image(
    image_moving_shape[0], grid_spacing, line_thickness)

image_moving = np.moveaxis(image_moving, -1, 0)
image_moving = al.utils.image_from_numpy(
    image_moving, [1, 1], [0, 0], dtype=dtype, device=device)

transformation = al_transformation.pairwise.BsplineTransformation(image_moving.size,
                                                                  sigma=sigma,
                                                                  rgb=False,
                                                                  order=1,
                                                                  dtype=dtype,
                                                                  device=device,
                                                                  diffeomorphic=True)

p = transformation.trans_parameters.detach()
p[0, 0, 4, 4] = 0.2
p[0, 1, 4, 4] = 0.
transformation.trans_parameters = Parameter(p)

displacement = transformation.get_displacement()
warped_image = al_transformation.utils.warp_image(
    image_moving, displacement)

new_image = warped_image.image.detach().cpu().numpy().squeeze()

keypoints = np.array([[2000.0, 1800.0], [80.0, 10.0]])

tmp_displacement1 = al_transformation.utils.unit_displacement_to_displacement(
    displacement)  # unit measures to image domain measures
tmp_displacement2 = al.create_displacement_image_from_image(
    displacement, image_moving)
deformed_landmarks = al.utils.points.Points.transform(
    keypoints, tmp_displacement2)

plt.close()
fig = plt.figure()
plt.imshow(new_image)
plt.scatter(keypoints[:, 0], keypoints[:, 1], c='b',
            marker='x', label='Original Keypoints')
plt.scatter(deformed_landmarks[:, 0], deformed_landmarks[:, 1],
            c='g', marker='x', label='Deformed Landmarks')
plt.legend()
plt.show()
plt.savefig(
    "/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/grid.png")

print(transformation.trans_parameters.shape)

x = 0
