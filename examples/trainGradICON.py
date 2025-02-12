from pathlib import Path
import sys
import builtins

from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import random
import icon_registration
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers

original_input = builtins.input


def fixed_input() -> str:
    return "gradicon_fixed"


builtins.input = fixed_input

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.training.train_voxelmorph import VoxelMorph  # nopep8
from registrationbaselines.data_loading.data_loaders import L2RLungCTDataset  # nopep8
from registrationbaselines.core import utils, utils_dl  # nopep8

GPUS = 1
BATCH_SIZE = 1


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


def main():

    idxs = np.arange(5)
    # np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:1], idxs[:1]

    base_dir = Path(__file__).parent.parent.absolute()

    data_path = Path("/home/koeglf/data/LungCT/LungCT_preprcoessed/")

    train_dataset = L2RLungCTDataset(dataset_path=data_path,
                                     indices=list(train_idx),
                                     return_type="torch_tensor_dict")
    val_dataset = L2RLungCTDataset(dataset_path=data_path,
                                   indices=list(val_idx),
                                   return_type="torch_tensor_dict")

    print("train dataset:", len(train_dataset),
          "val dataset:  ", len(val_dataset))

    # vxm_config_file = base_dir / "registrationbaselines/configs/Voxelmorph.yaml"
    # vxm_training = VoxelmorphTraining(
    #     train_dataset, vxm_config_file, val_dataset)

    hires_net = make_network([1, 1] + list(train_dataset.image_shape))

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()
    optimizer = torch.optim.Adam(
        (p for p in net_par.parameters() if p.requires_grad), lr=0.00005
    )

    net_par.train()

    dataloader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    def make_batch() -> Tuple[torch.Tensor, torch.Tensor]:

        item = random.choice(train_dataset)

        fixed_image = item['fixed_image'].cuda()
        moving_image = item['moving_image'].cuda()

        fixed_image = utils_dl.reshape_tensor(
            fixed_image, [4 * 40, 4 * 96, 4 * 96])
        moving_image = utils_dl.reshape_tensor(
            moving_image, [4 * 40, 4 * 96, 4 * 96])

        fixed_image.unsqueeze_(0).unsqueeze_(0)
        moving_image.unsqueeze_(0).unsqueeze_(0)

        return moving_image, fixed_image

    icon_registration.train_batchfunction(net_par,
                                          optimizer,
                                          make_batch,
                                          unwrapped_net=hires_net,
                                          steps=11)


if __name__ == "__main__":
    main()
