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
    img = np.ones((size[0], size[1]), dtype=np.uint8) * 255

    # Draw horizontal lines
    for y in range(0, size[0], grid_size):
        img[y:y+line_thickness, :] = 0  # Set the rows to black

    # Draw vertical lines
    for x in range(0, size[1], grid_size):
        img[:, x:x+line_thickness] = 0  # Set the columns to black

    return img


image_moving_shape = (20, 40)
grid_spacing = 2
line_thickness = 5
dtype = torch.float32
device = torch.device("cuda:0")


# Example usage
image = create_grid_image(
    image_moving_shape, grid_spacing, line_thickness)

image = al.utils.image_from_numpy(
    image, [1, 1], [0, 0], dtype=dtype, device=device)

for i in range(5):
    image_save = image.itk()
    sitk.WriteImage(image_save,
                    '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/im_rev.nii.gz')
    image_load = sitk.ReadImage(
        '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results/im_rev.nii.gz')
    print(f"shape itk directly loaded:\t{image_load.GetSize()}")

    image_load = al.create_tensor_image_from_itk_image(
        image_load, device="cuda:0")

    image_airlab_shape = image.image.cpu().numpy().squeeze().shape
    image_itk_shape = image_load.image.cpu().numpy().squeeze().shape

    image = image_load

    print(f"shape airlab image:\t{image_airlab_shape}")
    print(f"shape sitk image:\t{image_itk_shape}\n\n")
