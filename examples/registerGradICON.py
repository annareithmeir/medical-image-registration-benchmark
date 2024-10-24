from pathlib import Path
import sys

from typing import Tuple, List

import icon_registration.itk_wrapper as itk_wrapper
import icon_registration.pretrained_models as pretrained_models
import itk
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import random
import icon_registration
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset  # nopep8


def make_network(shape: List[int]):

    phi = network_wrappers.FunctionFromVectorField(
        networks.tallUNet2(dimension=3)
    )
    psi = network_wrappers.FunctionFromVectorField(
        networks.tallUNet2(dimension=3))

    hires_net = icon_registration.GradientICON(
        network_wrappers.DoubleNet(
            network_wrappers.DownsampleNet(
                network_wrappers.TwoStepRegistration(phi, psi), dimension=3
            ),
            network_wrappers.FunctionFromVectorField(
                networks.tallUNet2(dimension=3)),
        ),
        icon_registration.LNCCOnlyInterpolated(sigma=5),
        3,
    )
    hires_net.assign_identity_map([1, 1, 4 * 40, 4 * 96, 4 * 96])
    return hires_net


# model = pretrained_models.OAI_knees_gradICON_model()  # GradICON
# model = pretrained_models.OAI_knees_registration_model() #ICON


data_path = Path("/data/LungCT_preprocessed_new")
dataset = L2RLungCTDataset(dataset_path=data_path,
                           indices=[0],
                           return_type="path_dict")

model = make_network([1, 1] + list(dataset.image_shape))

item = dataset[0]
fixed_image_path = item['fixed_image']
moving_image_path = item['moving_image']

# Feel free to experiment with different images here
image_A = itk.imread(fixed_image_path)
image_B = itk.imread(moving_image_path)

phi_AB, phi_BA = itk_wrapper.register_pair(model, image_A, image_B)

x = 0
