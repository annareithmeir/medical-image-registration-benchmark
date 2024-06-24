import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def features_pca(features: np.ndarray, num_dims: int) -> np:
    pca = PCA(n_components=num_dims)
    pca.fit(features)
    pca_features = pca.transform(features)
    return pca_features


path_image = "registrationbaselines/examples/visualisation_scripts/visualise_features/image_dino.pt"
path_features_dino = "registrationbaselines/examples/visualisation_scripts/visualise_features/features_dino.pt"
path_features_sam = "registrationbaselines/examples/visualisation_scripts/visualise_features/features_sam.pt"
path_features_medsam = "registrationbaselines/examples/visualisation_scripts/visualise_features/features_medsam.pt"

image = torch.load(path_image, map_location='cpu').detach(
).numpy().squeeze()[0, :, :]
features_dino = torch.load(
    path_features_dino, map_location='cpu').detach().numpy()
features_sam = torch.load(
    path_features_sam, map_location='cpu').detach().numpy().transpose(1, 0)
features_medsam = torch.load(
    path_features_medsam, map_location='cpu').detach().numpy().transpose(1, 0)


# ========================================
# ========================================
# PARAMS
PCA_DIM = 5
# ========================================
# ========================================

features_pca_dino = features_pca(features_dino, PCA_DIM)
features_pca_sam = features_pca(features_sam, PCA_DIM)
features_pca_medsam = features_pca(features_medsam, PCA_DIM)


features_plot_dino = features_pca_dino.reshape(64, 64, PCA_DIM)
features_plot_sam = features_pca_sam.reshape(64, 64, PCA_DIM)
features_plot_medsam = features_pca_medsam.reshape(64, 64, PCA_DIM)

all_features = [features_plot_dino, features_plot_sam, features_plot_medsam]


# Create a figure with specified layout
plt.close('all')
font_size = 16
fig, axes = plt.subplots(3, PCA_DIM + 1, figsize=((PCA_DIM + 1)*3, 9))

titles_features = [f"PCA dim {dim}/{PCA_DIM}" for dim in range(1, PCA_DIM + 1)]

all_features = [features_plot_dino, features_plot_sam, features_plot_medsam]


# Plot the middle slice of the image in the first column of the first row
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Middle slice", fontsize=font_size)

for i, feature_map in enumerate(all_features):
    for col in range(1, PCA_DIM + 1):
        axes[i, col].imshow(feature_map[:, :, col - 1])
        if i == 0:
            axes[i, col].set_title(
                titles_features[col - 1], fontsize=font_size)
        axes[i, col].axis('off')

# Add row titles rotated by 90 degrees on the very right
row_titles = ["DINOv2", "SAM", "MedSAM"]
for row, title in enumerate(row_titles):
    axes[row, -1].text(1.05, 0.5, title, va='center', ha='center',
                       rotation=-90, transform=axes[row, -1].transAxes, fontsize=font_size)


# Hide the first column in the second and third rows
for row in range(0, 3):
    axes[row, 0].axis('off')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()

plt.savefig(
    "registrationbaselines/examples/visualisation_scripts/visualise_features/features.png",
    bbox_inches='tight',
    pad_inches=0.03,
    dpi=300)

x = 0
