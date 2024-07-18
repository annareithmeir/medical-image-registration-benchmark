import torch
import numpy as np


def normalize_tensor_to_0_1(tensor: torch.tensor)-> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def check_isotropic_and_identity(subject):
    for image_name, image in subject.items():
        # Get the affine matrix
        affine = image.affine
        assert np.abs(affine[0,0]) == np.abs(affine[1,1]) == np.abs(affine[2,2]) # isotropic
        assert np.abs(affine[0,3]) == np.abs(affine[1,3]) == np.abs(affine[2,3]) # origin