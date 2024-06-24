from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8

from registrationbaselines.data_loading import data_loaders  # nopep8


path_data = Path("/data/ACDC/database/")
loader_data = data_loaders.ACDCDataset(path_data,
                                       return_mode="test_imgs4",
                                       normalize_mode=True,
                                       roi_only=True,
                                       dim_mode='2d-middle')

item = loader_data[1]

image_systol = item['img_x'][1].squeeze()
image_diastol = item['img_y'][1].squeeze()
segmentation_systol = item['labels_x'][1].squeeze()
segmentation_diastol = item['labels_y'][1].squeeze()

# Mask the segmentation arrays where values are 0
masked_segmentation_systol = np.ma.masked_where(
    segmentation_systol == 0, segmentation_systol)
masked_segmentation_diastol = np.ma.masked_where(
    segmentation_diastol == 0, segmentation_diastol)

colors = ['red', 'green', 'blue']
cmap = ListedColormap(colors)

# plot both images next to each other
plt.close()

seg_alpha = 0.4

fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
ax[0].imshow(image_systol, cmap='gray')
ax[0].imshow(masked_segmentation_systol, cmap=cmap,
             alpha=seg_alpha, vmin=0.1, vmax=4)
ax[0].set_title('End systolic phase', fontsize=18)
ax[0].axis('off')  # turn off axes for the second subplot

ax[1].imshow(image_diastol, cmap='gray')
ax[1].imshow(masked_segmentation_diastol,
             cmap=cmap, alpha=seg_alpha, vmin=0.1, vmax=4)
ax[1].set_title('End diastolic phase', fontsize=18)
ax[1].axis('off')  # turn off axes for the second subplot

legend_elements = [
    Patch(facecolor='red', edgecolor='none', label='Right ventricular cavity'),
    Patch(facecolor='green', edgecolor='none', label='Myocardium'),
    Patch(facecolor='blue', edgecolor='none', label='Left ventricular cavity')
]
fig.legend(handles=legend_elements, loc='upper center',
           fontsize=10, ncol=1, bbox_to_anchor=[0.5, 0.95], facecolor='lightgray', framealpha=1.0)
# ,
plt.tight_layout()

plt.show()

plt.savefig(
    'registrationbaselines/examples/visualisation_scripts/visualise_acdc/acdc_vis.png',
    bbox_inches='tight',
    pad_inches=0)
