import nibabel as nib
from scipy.interpolate import griddata
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.core import utils_nifti

import matplotlib.pyplot as plt


def load_correspondences(file_path):
    """ Load correspondence data from a binary file. """
    return np.fromfile(file_path, dtype=np.float32)


def reshape_data(corr_data):
    """ Reshape flat correspondence data into a structured format. """
    num_correspondences = len(corr_data) // 6
    return corr_data.reshape((num_correspondences, 6))


def prepare_interpolation_data(structured_data):
    """ Extract positions and compute displacements. """
    positions = structured_data[:, :3]
    displacements = structured_data[:, 3:] - structured_data[:, :3]
    return positions, displacements


def interpolate_displacements(positions, displacements, image_shape):
    """ Interpolate sparse displacements to a full 3D displacement field. """
    grid_x, grid_y, grid_z = np.mgrid[
        0:image_shape[0], 0:image_shape[1], 0:image_shape[2]
    ]
    return griddata(
        positions[:, [1, 0, 2]],  # Reorder y, x to x, y
        displacements,
        (grid_x, grid_y, grid_z),
        method='linear'
    )


def visualize_displacement_slice(displacement_field, component=0):
    """ Visualize a specific component of the displacement field. """
    mid_slice = displacement_field.shape[1] // 2
    plt.figure(figsize=(10, 7))
    plt.imshow(displacement_field[:, mid_slice, :,
               component], cmap='jet', origin='lower')
    plt.colorbar()
    plt.title(
        f'{["X", "Y", "Z"][component]}-Component of Displacement Field at Middle Y-Slice')
    plt.xlabel('Z-dimension')
    plt.ylabel('X-dimension')
    plt.show()


def main():
    # Path to the uploaded correspondence file
    file_path = r"/home/fryderyk/Documents/data/results/DeformableCorrField/deformations/tumor2_resampled111_normalized_deformation_to_tumor1_resampled111_normalized.dat"

    image_shape = (240, 240, 157)  # x, y, z dimensions of the image

    corr_data = load_correspondences(file_path)
    structured_data = reshape_data(corr_data)
    positions, displacements = prepare_interpolation_data(structured_data)

    displacement_field = interpolate_displacements(
        positions, displacements, image_shape)

    fixed_image = nib.load(
        "registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz")

    # create a new image with the displacement field using the same header as the fixed image
    displacement_field_image = nib.Nifti1Image(
        displacement_field, fixed_image.affine)
    nib.save(displacement_field_image, "displacement_field.nii.gz")

    utils_nifti.set_intent_code(
        Path("displacement_field.nii.gz"), 'NIFTI_INTENT_DISPVECT')

    x = 0


if __name__ == '__main__':
    main()
