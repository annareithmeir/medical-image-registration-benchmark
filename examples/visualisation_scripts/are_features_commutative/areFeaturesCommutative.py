import torch
import sys
import os
from pathlib import Path


import matplotlib.pyplot as plt
import torch as th
import numpy as np
from PIL import Image
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent.parent / "latent_space_registration"))  # nopep8


import latent_space_registration.airlab as al
import latent_space_registration.airlab.transformation as transformation


def features_pca(features: np.ndarray, num_dims: int) -> np:
    pca = PCA(n_components=num_dims)
    pca.fit(features)
    pca_features = pca.transform(features)
    return pca_features


def deform_image(dtype, device, moving_image, sigma: list):

    # define the transformation
    transformation = al.airlab.transformation.pairwise.BsplineTransformation(moving_image.size,
                                                                             sigma=sigma,
                                                                             order=1,
                                                                             dtype=dtype,
                                                                             device=device,
                                                                             rgb=False)

    # change values of transformation.trans_parameters
    trans_parameters_clone = transformation.trans_parameters.clone()
    trans_parameters_clone[0, 0, 3, 2] = 0.2
    trans_parameters_clone[0, 1, 3, 2] = 0.2
    transformation.trans_parameters = torch.nn.Parameter(
        trans_parameters_clone)

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.airlab.transformation.utils.warp_image(
        moving_image, displacement)

    return warped_image


path_image = "registrationbaselines/examples/visualisation_scripts/visualise_features/image_sam.pt"
path_features = "registrationbaselines/examples/visualisation_scripts/visualise_features/features_sam.pt"

image = torch.load(path_image, map_location='cpu').detach(
).numpy().squeeze()[0, :, :]
image = al.utils.image_from_numpy(image, [1, 1], [0, 0], device='cpu')

image_size = image.size

features = torch.load(
    path_features, map_location='cpu').transpose(1, 0)
feature_dim1 = features_pca(features.detach().cpu(),
                            5).squeeze()[:, 2].reshape(64, 64)
feature_dim1 = F.interpolate(torch.Tensor(feature_dim1, device='cpu').unsqueeze(0).unsqueeze(0),
                             scale_factor=16,
                             mode='bilinear').squeeze()
feature_dim1 = al.utils.image_from_numpy(feature_dim1.detach().cpu().numpy().reshape(image_size),
                                         [1, 1],
                                         [0, 0],
                                         device='cpu')

image_deformed = deform_image(th.float32, 'cpu', image, sigma=[300, 300])


# DONE HERE enocde image, deform feature
feature_dim1_deformed = deform_image(
    th.float32, 'cpu', feature_dim1, sigma=[300, 300])

features_of_image_deformed = torch.load("/u/home/koeglf/Documents/code/registrationbaselines/examples/visualisation_scripts/are_features_commutative/features_sam_deformed.pt",
                                        map_location='cpu').detach().numpy().transpose(1, 0)
features_of_image_deformed_dim = features_pca(
    features_of_image_deformed, 5).squeeze()[:, 2].reshape(64, 64)
features_of_image_deformed_dim = F.interpolate(torch.Tensor(features_of_image_deformed_dim, device='cpu').unsqueeze(0).unsqueeze(0),
                                               scale_factor=16,
                                               mode='bilinear').squeeze()

# # DONE HERE deform image, encode image
# features_of_image_deformed_dim1 = al.utils.image_from_numpy(features_of_image_deformed_dim1.detach().cpu().numpy().reshape(image_size),
#                                                             [1, 1],
#                                                             [0, 0],
#                                                             dtype=th.float32,
#                                                             device='cpu')


# save the images
base_path = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/copy'

image_middle = image.numpy().squeeze()
image_middle_feature = feature_dim1.numpy()
image_encode_then_deform = feature_dim1_deformed.numpy()
image_deform_then_encode = features_of_image_deformed_dim.numpy()


def min_max_normalisation(image: np.ndarray):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


image_middle = min_max_normalisation(image_middle)
image_middle_feature = min_max_normalisation(image_middle_feature)
image_encode_then_deform = min_max_normalisation(image_encode_then_deform)
image_deform_then_encode = min_max_normalisation(image_deform_then_encode)
image_difference = min_max_normalisation(
    image_encode_then_deform - image_deform_then_encode)


plt.imshow(image_middle, cmap='gray')
plt.axis('off')
plt.savefig(os.path.join(base_path, 'middle.png'),
            bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.imshow(image_middle_feature)
plt.axis('off')
plt.savefig(os.path.join(base_path, 'middle_feature.png'),
            bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.imshow(image_encode_then_deform)
plt.axis('off')
plt.savefig(os.path.join(base_path, 'encode_then_deform.png'),
            bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.imshow(image_deform_then_encode)
plt.axis('off')
plt.savefig(os.path.join(base_path, 'deform_then_encode.png'),
            bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.imshow(image_difference, cmap='gray')
plt.axis('off')
plt.savefig(os.path.join(base_path, 'difference.png'),
            bbox_inches='tight', pad_inches=0.0, dpi=300)
