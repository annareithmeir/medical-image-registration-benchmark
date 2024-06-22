import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

# =============================================================================
# =============================================================================
# =============================================================================
# Load the provided CSV file

file_path = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/results_test/BSplines_no_encoder_MSE_lr0.0005_reg[256]_it[1500]_sigma[[6, 6]]/results.csv'

# =============================================================================
# =============================================================================
# =============================================================================

# print all rows
pd.set_option('display.max_rows', None)
df = pd.read_csv(file_path)[:-4]

# Melt the DataFrame to long format
df_melted = df.melt(value_vars=['dice_0', 'dice_1', 'dice_2', 'dice_mean'],
                    var_name='Metric', value_name='Value')


# Define colors for each boxplot
palette = ['#1f77b4', '#51a5e1', '#93cdf6', '#b0b0b0']

line_width = 1.7
# Create the boxplot using seaborn
plt.figure(figsize=(7, 6))

# Increase the line thickness of the figure
plt.rcParams['axes.linewidth'] = line_width

ax = sns.boxplot(
    data=df_melted, x="Metric", y="Value",
    notch=True, showcaps=True,
    flierprops={"marker": "o"},
    palette=palette,
    width=0.3,  # Increase the width of the boxes
    # whis=1.5,  # Increase the whisker length
    # dodge=True  # Increase the horizontal distance between boxplots
    linewidth=line_width,  # Increase the line width of the boxes
    medianprops={"color": "#b00202",
                 "linestyle": "-",
                 "linewidth": line_width*1.5}
)

x_labels = ['RV', 'LV-Myo', 'LV-BP', 'Mean']
ax.set_xticklabels(x_labels)

# Increase the line width of the ticks
plt.tick_params(axis='both', which='both', width=line_width)

# Customize the plot to match the style
plt.ylabel('DICE score', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)
plt.xlabel('')  # Remove the x-label

# set upper y limit to 1
plt.ylim(None, 1)


# Display the plot
plt.show()

plt.savefig(
    '/u/home/koeglf/Documents/code/registrationbaselines/examples/visualisation_scripts/boxplot.png')
