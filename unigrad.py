from unigradicon import get_unigradicon
import torch.nn.functional as F
import torch
import numpy as np
import itk
import matplotlib.pyplot as plt


################################################################
################################################################
"""
You have to change 
input_shape = [1, 1, 286, 235, 235]
in __init__ to match your image
"""
################################################################
################################################################

target_path = "/home/koeglf/Documents/fixed.nii.gz"
source_path = "/home/koeglf/Documents/moving.nii.gz"

target_itk = itk.imread(target_path)
target_meta = dict(target_itk)
target = np.asarray(target_itk)
print(f"Target shape: {target.shape}")
print(f"Target spacing: {target_meta['spacing']}")
print(f"Target direction: {target_meta['direction']}")

source_itk = itk.imread(source_path)
source_meta = dict(source_itk)
source = np.asarray(source_itk)
print(f"Source shape: {source.shape}")
print(f"Source spacing: {source_meta['spacing']}")
print(f"Source direction: {source_meta['direction']}")

# Check whether the orientation of the images are the same.
assert np.array_equal(dict(target_itk)["direction"], dict(source_itk)[
                      "direction"]), "The orientation of source and target images need to be the same."

# Processing images


def preprocess(img, type="ct"):
    if type == "ct":
        # return img
        clamp = [-1000, 1000]
        # img = (torch.clamp(img, clamp[0], clamp[1]
        #                    ) - clamp[0])/(clamp[1]-clamp[0])
        return F.interpolate(img, [286, 235, 235], mode="trilinear", align_corners=False)
    elif type == "mri":
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)
    else:
        print(f"Error: Do not support the type {type}")
        return img


target = preprocess(torch.Tensor(
    np.array(target)).unsqueeze(0).unsqueeze(0), type="ct")
source = preprocess(torch.Tensor(
    np.array(source)).unsqueeze(0).unsqueeze(0), type="ct")

net = get_unigradicon()
net.cuda()
net.eval()
print()

with torch.no_grad():
    net(source.cuda(), target.cuda())

    source_warped = net.warped_image_A.cpu().squeeze().numpy()

    # expand to range [-1000, 1000]
    source_warped = source_warped * 2000 - 1000

    # save image with itk
    itk.imwrite(itk.GetImageFromArray(source_warped),
                "/home/koeglf/Documents/moving_warped.nii.gz")

    source_shift = source.squeeze().numpy() * 2000 - 1000
    itk.imwrite(itk.GetImageFromArray(source_shift),
                "/home/koeglf/Documents/moving_shift.nii.gz")

    target_shift = target.squeeze().numpy() * 2000 - 1000
    itk.imwrite(itk.GetImageFromArray(target_shift),
                "/home/koeglf/Documents/fixed_shift.nii.gz")

    net.warped_image_A
    net.phi_AB_vectorfield.cpu()

    x = 0
