import copy
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
# =============================================================================
# =============================================================================
# =============================================================================
# Load the provided CSV file

base_dir = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/'

path_zero_disp = base_dir + r'results_zero_displacement_test.csv'

path_ncc = base_dir + \
    r'results_set2/BSplines_no_encoder_1.0_NCC__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'

path_medsam_cosine = base_dir + \
    r'results_set2/BSplines_feat_MedSAM_1.0_COSINE__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'
path_medsam_l1 = base_dir + \
    r'results_set2/BSplines_feat_MedSAM_1.0_L1__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'

path_sam_cosine = base_dir + \
    r'results_set2/BSplines_feat_SAM_1.0_COSINE__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'
path_sam_l1 = base_dir + \
    r'results_set2/BSplines_feat_SAM_1.0_L1__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'

path_dinov2_cosine = base_dir + \
    r'results_set2/BSplines_feat_DINOv2_1.0_COSINE__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'
path_dinov2_l1 = base_dir + \
    r'results_set2/BSplines_feat_DINOv2_1.0_L1__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'

metric_mean = 'hausdorff_mean'
y_label = 'Hausdorff distance (px)'
# metric_mean = 'dice_mean'
# y_label = 'DICE score'
# metric_mean = 'frac_foldings'
# y_label = 'Fraction of foldings'
metric_mean = 'sdlogj'
y_label = 'Log Jacobian determinant (std)'

# =============================================================================
# =============================================================================
# =============================================================================

# print all rows
pd.set_option('display.max_rows', None)
df_zero_disp = pd.read_csv(path_zero_disp)[:-4]
df_mse = pd.read_csv(path_ncc)[:-4]
df_medsam = pd.read_csv(path_medsam)[:-4]
df_sam = pd.read_csv(path_sam)[:-4]
df_dinov2 = pd.read_csv(path_dinov2)[:-4]

# Extract only dice_mean from each DataFrame
df_zero_disp = df_zero_disp[[metric_mean]].rename(
    columns={metric_mean: 'Baseline'})
df_mse = df_mse[[metric_mean]].rename(columns={metric_mean: 'BSpline MSE'})
df_medsam = df_medsam[[metric_mean]].rename(
    columns={metric_mean: 'BSpline MedSAM'})
df_sam = df_sam[[metric_mean]].rename(columns={metric_mean: 'BSpline SAM'})
df_dinov2 = df_dinov2[[metric_mean]].rename(
    columns={metric_mean: 'BSpline DINOv2'})

vxm1 = copy.deepcopy(df_dinov2).rename(columns={'BSpline DINOv2': 'Vxm MSE'})
vxm2 = copy.deepcopy(df_dinov2).rename(
    columns={'BSpline DINOv2': 'Vxm DINOv2'})
vxm3 = copy.deepcopy(df_dinov2).rename(columns={'BSpline DINOv2': 'Vxm SAM'})
vxm4 = copy.deepcopy(df_dinov2).rename(
    columns={'BSpline DINOv2': 'Vxm MedSAM'})

if 'frac_foldings' in metric_mean or 'sdlogj' in metric_mean:
    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat([df_mse, df_dinov2, df_sam,
                    df_medsam, vxm1, vxm2, vxm3, vxm4], axis=1)

    # Melt the DataFrame to long format
    df_melted = df.melt(value_vars=['BSpline MSE',
                                    'BSpline DINOv2',
                                    'BSpline SAM',
                                    'BSpline MedSAM',
                                    'Vxm MSE',
                                    'Vxm DINOv2',
                                    'Vxm SAM',
                                    'Vxm MedSAM'],
                        var_name='Metric', value_name='Value')
else:
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
palette = ['#ffab2e', '#6e4fb0', '#2298ec', '#93cdf6',
           '#ffab2e', '#6e4fb0', '#2298ec', '#93cdf6']
if 'frac_foldings' not in metric_mean and 'sdlogj' not in metric_mean:
    palette = ['#b0b0b0'] + palette

line_width = 1.7
# Create the boxplot using seaborn
plt.figure(figsize=(15, 6))

# Increase the line thickness of the figure
plt.rcParams['axes.linewidth'] = line_width

ax = sns.boxplot(
    data=df_melted, x="Metric", y="Value",
    notch=True if 'frac_foldings' not in metric_mean and 'sdlogj' not in metric_mean else False,
    showcaps=True,
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

# Increase the line width of the ticks
plt.tick_params(axis='both', which='both', width=line_width)


# Customize the plot to match the style
plt.ylabel(y_label, fontsize=18)
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
