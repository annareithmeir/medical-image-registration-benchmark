import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
import torch.nn.functional as F
import torch
# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8

from registrationbaselines.core import utils  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8


def compute_grid(image_size, dtype=torch.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")


def warp_image(image, displacement):

    image_size = image.shape[-2:]

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)

    # warp image

    warped_image = F.grid_sample(image, - displacement + grid)

    return warped_image


path_data = Path("/data/ACDC/database/")
train_dataset = data_loaders.ACDCDataset(path_data, return_mode="train_imgs2", normalize_mode=True, roi_only=True,
                                         dim_mode='2d-middle')
x, _ = train_dataset.__getitem__(0)

disp = np.zeros((128, 128, 2))
# disp[80:100, 80:100, :] = 0


x = torch.from_numpy(x).unsqueeze(0)

x_warped = warp_image(x, torch.from_numpy(disp).unsqueeze(0))

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
plt.imshow(x.squeeze() - x_warped.squeeze())
plt.colorbar()
axes.axis("on")
fig.show()

plt.savefig(
    "/u/home/koeglf/Documents/code/registrationbaselines/examples/deformed_sitk.png")
plt.close()
