import copy
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
# =============================================================================
# =============================================================================
# =============================================================================
# Load the provided CSV file
path_zero_disp = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/results_zero_displacement_test.csv'
path_mse = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/results_test/BSplines_no_encoder_MSE_lr0.0005_reg[256]_it[1500]_sigma[[6, 6]]/results.csv'
path_medsam = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/BSplines_feat_val/BSplines_feat_MedSAM_COSINE_lr0.0005_reg[256]_it[1500]_sigma[[6, 6]]/results.csv'
path_sam = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/BSplines_feat_val/BSplines_feat_SAM_COSINE_lr0.0005_reg[256]_it[1500]_sigma[[6, 6]]/results.csv'
path_dinov2 = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/BSplines_feat_val/BSplines_feat_DINOv2_COSINE_lr0.0005_reg[256]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'

# =============================================================================
# =============================================================================
# =============================================================================

# print all rows
pd.set_option('display.max_rows', None)
df_zero_disp = pd.read_csv(path_zero_disp)[:-4]
df_mse = pd.read_csv(path_mse)[:-4]
df_medsam = pd.read_csv(path_medsam)[:-4]
df_sam = pd.read_csv(path_sam)[:-4]
df_dinov2 = pd.read_csv(path_dinov2)[:-4]

# Extract only dice_mean from each DataFrame
df_zero_disp = df_zero_disp[['dice_mean']].rename(
    columns={'dice_mean': 'Baseline'})
df_mse = df_mse[['dice_mean']].rename(columns={'dice_mean': 'BSpline MSE'})
df_medsam = df_medsam[['dice_mean']].rename(
    columns={'dice_mean': 'BSpline MedSAM'})
df_sam = df_sam[['dice_mean']].rename(columns={'dice_mean': 'BSpline SAM'})
df_dinov2 = df_dinov2[['dice_mean']].rename(
    columns={'dice_mean': 'BSpline DINOv2'})

vxm1 = copy.deepcopy(df_dinov2).rename(columns={'BSpline DINOv2': 'Vxm MSE'})
vxm2 = copy.deepcopy(df_dinov2).rename(
    columns={'BSpline DINOv2': 'Vxm DINOv2'})
vxm3 = copy.deepcopy(df_dinov2).rename(columns={'BSpline DINOv2': 'Vxm SAM'})
vxm4 = copy.deepcopy(df_dinov2).rename(
    columns={'BSpline DINOv2': 'Vxm MedSAM'})

# Concatenate all DataFrames into a single DataFrame
df = pd.concat([df_zero_disp, df_mse, df_dinov2, df_sam,
               df_medsam, vxm1, vxm2, vxm3, vxm4], axis=1)

# Melt the DataFrame to long format
df_melted = df.melt(value_vars=['Baseline',
                                'BSpline MSE',
                                'BSpline DINOv2',
                                'BSpline SAM',
                                'BSpline MedSAM',
                                'Vxm MSE',
                                'Vxm DINOv2',
                                'Vxm SAM',
                                'Vxm MedSAM'],
                    var_name='Metric', value_name='Value')

# =============================================================================
# =============================================================================
# =============================================================================

# Define colors for each boxplot
palette = ['#b0b0b0',
           '#ffab2e', '#6e4fb0', '#2298ec', '#93cdf6',
           '#ffab2e', '#6e4fb0', '#2298ec', '#93cdf6']

line_width = 1.7
# Create the boxplot using seaborn
plt.figure(figsize=(15, 6))

# Increase the line thickness of the figure
plt.rcParams['axes.linewidth'] = line_width

ax = sns.boxplot(
    data=df_melted, x="Metric", y="Value",
    notch=True, showcaps=True,
    flierprops={"marker": "o"},
    palette=palette,
    width=0.3,  # Adjust the width of the boxes
    linewidth=line_width,  # Increase the line width of the boxes
    medianprops={"color": "#b00202",
                 "linestyle": "-",
                 "linewidth": line_width*1.5},
    # Set the edge color of the boxes to black
    boxprops={"edgecolor": "black", "linewidth": line_width*0.7},
    # Set the whisker color to black
    whiskerprops={"color": "black", "linewidth": line_width*0.7},
    # Set the cap color to black
    capprops={"color": "black", "linewidth": line_width*0.7},
)

# =============================================================================
# =============================================================================
# =============================================================================
"""
x_lim = ax.get_xlim()
start = x_lim[0]  # -0.5
a = 0.5
b = 4.5
end = x_lim[1]  # 8.5

color_left = '#dedede'
color_middle = '#85d0ff'
color_right = '#fdbdff'

ax.axvspan(start, a, facecolor=color_left, alpha=0.3)  # Left
ax.axvspan(a, b, facecolor=color_middle, alpha=0.3)  # middle
ax.axvspan(b, end, facecolor=color_right, alpha=0.3)  # Right
ax.set_xlim(x_lim)

# Add legend for the face colors
legend_handles = [
    Patch(facecolor=color_left, edgecolor='none',
          alpha=0.3, label='No registration'),
    Patch(facecolor=color_middle, edgecolor='none',
          alpha=0.3, label='B-Spline registration'),
    Patch(facecolor=color_right, edgecolor='none',
          alpha=0.3, label='VoxelMorph registration')
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=12)
"""

# =============================================================================
# =============================================================================
# =============================================================================

# Increase the line width of the ticks
plt.tick_params(axis='both', which='both', width=line_width)


# Customize the plot to match the style
plt.ylabel('DICE score', fontsize=18)
plt.xticks(fontsize=16, rotation=45, ha='right')
plt.yticks(fontsize=14)
plt.xlabel('')  # Remove the x-label


# set upper y limit to 1
# plt.ylim(0, 1)

# =============================================================================
# =============================================================================
# =============================================================================
plt.tight_layout()
# Display the plot
plt.show()

plt.savefig(
    '/u/home/koeglf/Documents/code/registrationbaselines/examples/visualisation_scripts/boxplot.png')
