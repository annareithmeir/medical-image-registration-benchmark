import numpy as np
import torch
from typing import Tuple, Union, Optional
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from torchvision import transforms
import gc
import pandas as pd
import seaborn as sns
import torchio as tio
from segment_anything import sam_model_registry
from registrationbaselines.data_loading import data_loaders
from pathlib import Path
import sys
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent / "latent_space_registration"))

from latent_space_registration import FeatureExtractor


def load_encoder_model(name):
    if name == "DINOv2":
        model = torch.hub.load('facebookresearch/dinov2',
                               'dinov2_vitb14_reg',
                               pretrained=False)
        model_path = r"/home/anna/PycharmProjects/latent_space_registration/model_weights/dinov2_vitb14_reg4_pretrain.pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()

    elif name == "MedSAM":
        weights_path = "/home/anna/PycharmProjects/latent_space_registration/model_weights/medsam_vit_b.pth"
        model = sam_model_registry['vit_b'](checkpoint=weights_path)
        model.eval()

    elif name == "SAM":
        weights_path = "/home/anna/PycharmProjects/latent_space_registration/model_weights/sam_vit_b_01ec64.pth"
        model = sam_model_registry['vit_b'](checkpoint=weights_path)
        model.eval()

    else:
        model = None
        print("Not implemented")

    return model.to("cuda:0")


def _compute_MedSAM_features(model, image: torch.Tensor):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
    ])
    image = transform(image)

    image = image.unsqueeze(0)

    print("f", image.shape)

    features = model(image, multimask_output=False)

    features = features.view(256, 64 * 64)

    return features


def _compute_SAM_features(model, image: torch.Tensor):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
    ])
    image = transform(image)

    image = image.unsqueeze(0)

    print("f", image.shape)

    features = model(image)

    features = features.view(256, 64 * 64)

    return features


def _compute_DINOv2_features(self, image: np.ndarray | torch.Tensor, with_torch_no_grad: bool):
    """

    :param image: requires shape (h,w,3)
    :return:
    """
    # print("array type comp DINO()", type(image), image.dtype, image.shape, image.device)
    image_shape = image.shape[:2]
    # print("Before upsampling, image is of shape: ",image.shape)
    # shape_new = image.shape
    # shape_new = [(x + (14-x % 14))*UPSAMPLE_FACTOR for x in list(shape)]
    # print("After upsampling, image is of shape: ",shape_new)

    # image = Image.fromarray(image)

    UPSAMPLE_FACTOR = 7

    image = image.movedim(2, 0)
    transform1 = transforms.Compose([
        transforms.Resize(
            (image_shape[0] * UPSAMPLE_FACTOR, image_shape[1] * UPSAMPLE_FACTOR)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.2)
    ])

    image = transform1(image)
    total_features = []
    if with_torch_no_grad:
        with torch.no_grad():
            # print(next(self.encoder.parameters()).device)
            features_dict = self.encoder.forward_features(
                image.unsqueeze(0))
            features = features_dict['x_norm_patchtokens']
            total_features.append(features)
    else:
        features_dict = self.encoder.forward_features(image.unsqueeze(0))
        features = features_dict['x_norm_patchtokens']
        total_features.append(features)

    features = torch.cat(total_features, dim=0).squeeze()
    # print("DINOv2 Features shape:",features.shape)
    return features


def _reshape_DINOv2_features(self, features: torch.Tensor, image_shape: list[int]):
    # print("before",features.shape)
    patch_size = self.encoder.patch_size  # patchsize=14

    # patch_h = image_shape[0] // patch_size  # 37
    patch_h = image_shape[0]
    # patch_w = image_shape[1] // patch_size  # 37
    patch_w = image_shape[1]
    feat_dim = features.shape[-1]  # vitl14
    # print(patch_h, patch_w, patch_h * patch_w, patch_size)

    total_features = features.reshape(
        patch_h, patch_w, feat_dim)  # (*H*w, 1024)
    # print(total_features.shape)
    return total_features


def normalize_tensor_per_batch_sample(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
        Normalize each sample in a batch tensor to the range [0, 1].

        Parameters:
        batch_tensor (torch.Tensor): The input batch tensor to normalize.

        Returns:
        torch.Tensor: The batch tensor with each sample normalized.
        """
    # Initialize a tensor to store the normalized batch
    normalized_batch = torch.zeros_like(batch_tensor)

    # Iterate over the batch dimension
    for i in range(batch_tensor.size(0)):
        tensor = batch_tensor[i]
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        # Avoid division by zero
        if tensor_max > tensor_min:
            normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            normalized_tensor = tensor - tensor_min  # If all values are the same, return zero tensor

        normalized_batch[i] = normalized_tensor

    return normalized_batch


def compute_features_from_batch_of_RGB_images(name, model, image: torch.Tensor, with_torch_no_grad,
                                              reshape: Optional[bool] = False, normalize: bool = False):
    """

    :param expects torch.Tensor batch of 3-channel images according to torch convention image_batch: (bs, 3,h,w)
    :return: features of shape (bs,1024, h*w) or (bs, 1024, h,w)
    """

    if image.shape[0] == 1:
        Warning("Running function with grayscale data")
        image = image.repeat(3, 1, 1)

    print(image.shape)

    features_batch = []

    # for i in range(image_batch.shape[0]):
    #     image = image_batch[i]
    if name == "DINOv2":
        # image = (image * 255).type(torch.uint8).movedim(0,2)
        image = image.movedim(0, 2)
        features = _compute_DINOv2_features(model,
                                            image, with_torch_no_grad=with_torch_no_grad)
        if reshape:
            # features = _reshape_DINOv2_features(
            #     features, [image_batch.shape[2], image_batch.shape[3], 1024])
            # # features = self._reshape_DINOv2_features(features, [tmp* 14 for tmp in shape_new])
            # features_batch.append(torch.movedim(features, 2, 0))
            pass
        else:
            features_batch.append(torch.movedim(features, 1, 0))
    elif name == "MedSAM":
        features_batch.append(_compute_MedSAM_features(model, image.movedim(0, -1)))

    elif name == "SAM":
        features_batch.append(_compute_SAM_features(model, image))
    else:
        print("Not implemented")

    features_batch = torch.stack(features_batch)

    if normalize:
        features_batch = normalize_tensor_per_batch_sample(
            features_batch)

    return features_batch


def params_to_mat_2d(rotation: torch.Tensor, translation: torch.Tensor, scaling: torch.Tensor):
    cos = torch.cos(rotation)
    sin = torch.sin(rotation)
    rotation = torch.eye(2, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    rotation[:, 0, 0] = cos[:, 0]
    rotation[:, 0, 1] = -sin[:, 0]
    rotation[:, 1, 0] = sin[:, 0]
    rotation[:, 1, 1] = cos[:, 0]

    scaling = torch.diag_embed(scaling)
    rotation = torch.bmm(rotation, scaling)

    affines = torch.eye(3, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    affines[:, :2, :2] = rotation
    affines[:, :2, 2] = translation
    return affines


def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[1:-1])
        norm_factors = norm_factors.reshape(1, *(1,) * ndim, ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[1:-1], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, *(1,)*ndim, ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def warp(x: torch.Tensor, disp: torch.Tensor, start_coords=None, interp_mode="bilinear", normalise=False):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, channels, *sizes)) input image
        disp: (Tensor float, shape (N, *sizes, ndim)) dense disp field in h-w-d order
        interp_mode: (string) mode of interpolation in grid_sample()
        normalise: (bool) whether to normalise disp by the spatial sizes of the array

    Returns:
        deformed x, Tensor of the same shape as input
    """
    assert disp.shape[-1] in {2, 3}
    ndim = x.ndim - 2
    size = disp.shape[1:-1]
    disp = disp.type_as(x)

    if normalise:
        # # normalise disp to [-1, 1]
        disp = normalise_disp(disp.moveaxis(-1, 1)).moveaxis(1, -1)

    # generate standard mesh grid
    if start_coords is None:
        start_coords = torch.meshgrid([torch.linspace(-1, 1, size[i], device=x.device, dtype=x.dtype)
                               for i in range(ndim)], indexing="ij")
        start_coords = torch.stack(start_coords, dim=-1).tile((disp.shape[0], 1, *([1]*len(size))))
    warped_grid = start_coords + disp

    # grid_sample takes in x-y-z ordering, so we need to flip the coordinate dim
    warped_grid = warped_grid.flip(dims=(-1,))
    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=True, padding_mode="border")


def random_affine_displacements_2d(shape: Tuple[int],
                                   theta_range: Union[Tuple[float,], Tuple[float,]] = (0.,),
                                   trans_range: Union[Tuple[float, float], Tuple[float, float, float]] = (0., 0.),
                                   scale_range: Union[Tuple[float, float], Tuple[float, float, float]] = (0,0),
                                   ) -> torch.Tensor:
    if isinstance(theta_range, (float, int)):
        theta_range = [theta_range]
    if isinstance(trans_range, (float, int)):
        trans_range = [trans_range] * 2
    if isinstance(scale_range, (float, int)):
        scale_range = [scale_range] * 2
    if len(theta_range) > 1:
        raise ValueError("Give me 1 element, not more!")
    if len(trans_range) > 2:
        raise ValueError("Give me either 1 or 2 elements, not more!")
    if len(scale_range) > 2:
        raise ValueError("Give me either 1 or 2 elements, not more!")
    if len(theta_range) == 1:
        pass
    if len(trans_range) == 1:
        trans_range = [trans_range[0]] * 2
    if len(scale_range) == 1:
        scale_range = [scale_range[0]] * 2

    batch_size, channel_dim, spatial_dims = shape[0], shape[1], shape[2:]
    # Define randomly sampled affine params
    theta_range = torch.tensor(theta_range)
    trans_range = torch.tensor(trans_range)
    scale_range = torch.tensor(scale_range)
    # rotation = theta_range[None]
    rotation = torch.ones((shape[0], 1)) * theta_range * 2 - theta_range
    # translation = trans_range[None]
    translation = torch.ones((shape[0], 2)) * trans_range * 2 - trans_range
    scaling = 1 + torch.ones((shape[0], 2)) * scale_range * 2 - scale_range
    # Generate affine matrices
    affines = params_to_mat_2d(rotation, translation, scaling)
    # Define and augment coordinates
    coord_start = torch.meshgrid([torch.linspace(-1., 1., i) for i in spatial_dims], indexing="ij")
    aug_dim = torch.ones_like(coord_start[0])
    coord_start_aug = torch.stack([*coord_start, aug_dim], dim=-1)
    coord_start_aug = coord_start_aug.tile((batch_size, *[1]*len(coord_start_aug.shape)))
    coord_start_aug_ = coord_start_aug.reshape((batch_size, -1, coord_start_aug.shape[-1]))
    # Transform coordinates
    coord_end_aug_ = torch.bmm(coord_start_aug_, affines.transpose(1, 2))
    # coord_end_aug_ = torch.bmm(coord_start_aug_, affines.transpose(1,2))
    coord_end_aug = coord_end_aug_.reshape(coord_start_aug.shape)
    # Calculate displacement
    displ_aug = coord_end_aug - coord_start_aug
    # Remove aug dimension
    displ = displ_aug[..., :len(spatial_dims)]
    return displ


def apply_augs(fixedimg, angles=(0., ), translation=(0.,0.), scale=(0.,0.), do_affine=True, do_deform=False, intersubject=False):
    # Since we can't easily get the inverse of the displacement field, the deformed image becomes the
    # fixed image and we must learn to recreate the displacement field to be applied to untransformed moving image.
    batch_size, channels, *spatial_dims = fixedimg.shape
    num_spatial_dims = len(spatial_dims)
    disp = torch.zeros((batch_size, *spatial_dims, len(spatial_dims)))

    # disp_affine = random_affine_displacements_2d(num_spatial_dims, (batch_size, channels, *spatial_dims),
                                              # theta_range=angles, trans_range=translation, scale_range=scale)
    disp_affine = random_affine_displacements_2d(fixedimg.shape, theta_range=angles, trans_range=translation, scale_range=scale)
    disp += disp_affine

    return warp(fixedimg, disp, interp_mode='bilinear', normalise=False)


def COSINE(f1, f2):
    return - torch.mean(torch.nn.functional.cosine_similarity(f1, f2))


def L1(f1, f2):
    return torch.nn.L1Loss(reduce=True, reduction='mean')(f1, f2)



path_data = Path("/home/anna/datasets/ACDC")
loader_data = data_loaders.ACDCDataset(path_data,
                                   return_mode="test_imgs2",
                                   normalize_mode=True,
                                   roi_only=True,
                                   dim_mode='2d-middle')

y,x = loader_data.__getitem__(0)
y=torch.from_numpy(y).to("cuda").type(torch.float)

rot_range=(-math.pi/2, 1.2*math.pi/2)
# rot_step = math.pi/2
rot_step = math.pi/32
angle_ls = np.arange(rot_range[0], rot_range[1]+rot_step, rot_step)

tra_range = (-0.5, 0.5)
# tra_step = 0.5
tra_step = 0.01
tra_ls = np.arange(tra_range[0], tra_range[1]+tra_step, tra_step)

errs= { "SAM_L1_ls" : list(),
        "SAM_COSINE_ls":list(),
        "MedSAM_L1_ls" : list(),
        "MedSAM_COSINE_ls":list(),
        "DINOv2_L1_ls" : list(),
        "DINOv2_COSINE_ls":list()}

errs_tra= { "SAM_L1_ls" : list(),
        "SAM_COSINE_ls":list(),
        "MedSAM_L1_ls" : list(),
        "MedSAM_COSINE_ls":list(),
        "DINOv2_L1_ls" : list(),
        "DINOv2_COSINE_ls":list()}



for name in ["SAM", "MedSAM", "DINOv2"]:

    print(name)

    model = FeatureExtractor.FeatureExtractor(name)
    fy = model.compute_features_from_batch_of_RGB_images(y)

    if name =="SAM":
        angle_plots = np.arange(rot_range[0], rot_range[1], (rot_range[1]-rot_range[0])/8)
        fig, axes = plt.subplots(1,len(angle_plots)+1, figsize=(15, 8))
        i=0
        for rot_i in angle_plots:
            y_moved = apply_augs(y.clone().unsqueeze(0), angles=(float(rot_i),), translation=(0., 0.)).squeeze(0)
            axes[i].imshow(y_moved.cpu().squeeze().movedim(0, -1))
            i+=1
        plt.show()

    ### rotation #####
    tmp = 0
    for rot_i in angle_ls:
        # print(rot_i)
        y_moved = apply_augs(y.clone().unsqueeze(0), angles=(float(rot_i),), translation=(0., 0.)).squeeze(0)
        fm=model.compute_features_from_batch_of_RGB_images(y_moved)

        diff_l1 = L1(fy, fm)
        diff_cos = COSINE(fy, fm)

        errs[name+"_L1_ls"].append(float(diff_l1))
        errs[name+"_COSINE_ls"].append(float(diff_cos))

        tmp += 1


    if name =="SAM":
        tra_plots = np.arange(tra_range[0], tra_range[1], (tra_range[1]-tra_range[0])/8)
        fig, axes = plt.subplots(1,len(tra_plots)+1, figsize=(15, 8))
        i=0
        for tra_i in tra_plots:
            y_moved = apply_augs(y.clone().unsqueeze(0), angles=(0.0,), translation=(tra_i, )).squeeze(0)
            axes[i].imshow(y_moved.cpu().squeeze().movedim(0, -1))
            i+=1
        plt.show()


    ### translation #####
    tmp=0
    for tra_i in tra_ls:
        y_moved = apply_augs(y.clone().unsqueeze(0), angles=(0.0,), translation=(tra_i,)).squeeze(0)
        fm = model.compute_features_from_batch_of_RGB_images(y_moved)

        diff_l1 = L1(fy, fm)
        diff_cos = COSINE(fy, fm)

        errs_tra[name + "_L1_ls"].append(float(diff_l1))
        errs_tra[name + "_COSINE_ls"].append(float(diff_cos))

        tmp+=1



### plot ###

blues = ["#AED6E5", "#6CB4D0", "#51A6C8", "#378DAE"]
oranges = ["#FFC370", '#FFAC38', '#E17E2D', '#C96A1D']
purples = ["#D9B8FF", "#C08AFF", "#A04DFF", '#8214FF']

sns.set_theme()
sns.set_style(style='white')
labels = [r'$-\pi$', r'$-0.8\pi$', r'$-0.6\pi$', r'$-0.4\pi$',
          r'$-0.2\pi$', r'$0$', r'$0.2\pi$', r'$0.4\pi$', r'$0.6\pi$', r'$0.8\pi$', r'$\pi$']
labels2 = ["-90deg", "0deg", "90deg"]

labels_tra = ["-64px", "0", "64px"]

### per ror/tra ####

fig = plt.figure(figsize=(14,3))

data_sam_l1 = pd.DataFrame({'rotation': angle_ls, 'L1': errs["SAM_L1_ls"]})
data_sam_cosine = pd.DataFrame({'rotation': angle_ls, '-cosine': errs["SAM_COSINE_ls"]})
data_medsam_l1 = pd.DataFrame({'rotation': angle_ls, 'L1': errs["MedSAM_L1_ls"]})
data_medsam_cosine = pd.DataFrame({'rotation': angle_ls, '-cosine': errs["MedSAM_COSINE_ls"]})
data_dino_l1 = pd.DataFrame({'rotation': angle_ls, 'L1': errs["DINOv2_L1_ls"]})
data_dino_cosine = pd.DataFrame({'rotation': angle_ls, '-cosine': errs["DINOv2_COSINE_ls"]})

ax=fig.add_subplot(1,4,1)
sns.lineplot(data=data_sam_l1, x='rotation', y='L1', label='SAM', color=oranges[1], linestyle='dashed')
sns.lineplot(data=data_medsam_l1, x='rotation', y='L1', label='MedSAM', color=blues[1], linestyle='dashed')
sns.lineplot(data=data_dino_l1, x='rotation', y='L1', label='DINOv2', color=purples[1], linestyle='dashed')
plt.xticks(ticks=[rot_range[0], 0, rot_range[1]], labels=labels2)
plt.legend([],[], frameon=False)

ax=fig.add_subplot(1,4,2)
sns.lineplot(data=data_sam_cosine, x='rotation', y='-cosine', label='SAM', color=oranges[1])
sns.lineplot(data=data_medsam_cosine, x='rotation', y='-cosine', label='MedSAM', color=blues[1])
sns.lineplot(data=data_dino_cosine, x='rotation', y='-cosine', label='DINOv2', color=purples[1])

plt.xticks(ticks=[rot_range[0], 0, rot_range[1]], labels=labels2)
# plt.title("rotation")
plt.legend([],[], frameon=False)


data_sam_l1 = pd.DataFrame({'translation': tra_ls, 'L1': errs_tra["SAM_L1_ls"]})
data_sam_cosine = pd.DataFrame({'translation': tra_ls, '-cosine': errs_tra["SAM_COSINE_ls"]})
data_medsam_l1 = pd.DataFrame({'translation': tra_ls, 'L1': errs_tra["MedSAM_L1_ls"]})
data_medsam_cosine = pd.DataFrame({'translation': tra_ls, '-cosine': errs_tra["MedSAM_COSINE_ls"]})
data_dino_l1 = pd.DataFrame({'translation': tra_ls, 'L1': errs_tra["DINOv2_L1_ls"]})
data_dino_cosine = pd.DataFrame({'translation': tra_ls, '-cosine': errs_tra["DINOv2_COSINE_ls"]})

fig.add_subplot(1,4,3)
plt.xticks(ticks=[tra_range[0], 0, tra_range[1]], labels=labels_tra)
sns.lineplot(data=data_sam_l1, x='translation', y='L1', label='SAM', color=oranges[1], linestyle='dashed')
sns.lineplot(data=data_medsam_l1, x='translation', y='L1', label='MedSAM', color=blues[1], linestyle='dashed')
sns.lineplot(data=data_dino_l1, x='translation', y='L1', label='DINOv2', color=purples[1], linestyle='dashed')
plt.legend([],[], frameon=False)

fig.add_subplot(1,4,4)
sns.lineplot(data=data_sam_cosine, x='translation', y='-cosine', label='SAM', color=oranges[1])
sns.lineplot(data=data_medsam_cosine, x='translation', y='-cosine', label='MedSAM', color=blues[1])
sns.lineplot(data=data_dino_cosine, x='translation', y='-cosine', label='DINOv2', color=purples[1])
plt.xticks(ticks=[tra_range[0], 0, tra_range[1]], labels=labels_tra)


ax_list = fig.axes
ax=ax_list[-1]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 1, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.tight_layout()
plt.show()