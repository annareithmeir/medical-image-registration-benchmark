import numpy as np
import torch


def normalize_tensor_to_0_1(tensor: torch.tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def check_isotropic_and_identity(subject, scaling:int):
    for image_name, image in subject.items():
        # Get the affine matrix
        affine = image.affine
        # print(affine)
        # print(np.abs(affine[0, 0]), np.abs(affine[1, 1]) , np.abs(affine[2, 2]) , scaling)
        assert np.abs(affine[0, 0]) == np.abs(affine[1, 1]) == np.abs(affine[2, 2]) == scaling # isotropic
        assert np.abs(affine[1, 0]) == np.abs(affine[0, 1]) == np.abs(affine[2, 1]) == np.abs(affine[1, 2]) == np.abs(
            affine[3, 0]) == np.abs(affine[0, 3])==0  # off-diagonal zeros
        #assert np.abs(affine[0, 3]) == np.abs(affine[1, 3]) == np.abs(affine[2, 3])  # origin
