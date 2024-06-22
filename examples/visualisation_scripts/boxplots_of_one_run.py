import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
# =============================================================================
# Load the provided CSV file

file_path = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/results_test/BSplines_no_encoder_MSE_lr0.0005_reg[256]_it[1500]_sigma[[6, 6]]/results.csv'

# =============================================================================
# =============================================================================
# =============================================================================


df = pd.read_csv(file_path)

# Melt the DataFrame to long format
df_melted = df.melt(value_vars=['dice_0', 'dice_1', 'dice_2'],
                    var_name='Metric', value_name='Value')

# Define colors for each boxplot
palette = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create the boxplot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_melted, x="Metric", y="Value",
    notch=True, showcaps=False,
    flierprops={"marker": "x"},
    palette=palette,
)

# Customize the plot to match the style
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Boxplot of Dice Scores', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Set the y-axis limits to 0-1
plt.ylim(0, 1)

# Display the plot
plt.show()

plt.savefig(
    '/u/home/koeglf/Documents/code/registrationbaselines/examples/visualisation_scripts/boxplot.png')
