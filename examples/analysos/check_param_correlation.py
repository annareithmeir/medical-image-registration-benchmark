import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

import os
import pandas as pd
import re
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def extract_hyperparameters(folder_name):
    """
    Extract hyperparameters from folder name.
    """
    lr_match = re.search(r'lr([\d\.]+)', folder_name)
    reg_match = re.search(r'reg\[(.*?)\]', folder_name)
    sigma_match = re.search(r'sigma\[\[(.*?)\]\]', folder_name)

    lr = float(lr_match.group(1)) if lr_match else None
    reg = list(map(int, reg_match.group(1).split(', '))) if reg_match else None
    sigma = ast.literal_eval(
        f'[[{sigma_match.group(1)}]]') if sigma_match else None

    return lr, reg, sigma


def extract_final_rows_from_csv(file_path):
    """
    Extract the last four rows from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df.tail(4)


def main(base_dir):
    results = []

    # Traverse the directory structure
    for root, dirs, files in os.walk(base_dir):
        if "NCC" in root:
            for file in files:
                if file == 'results.csv':
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)
                    lr, reg, sigma = extract_hyperparameters(folder_name)
                    final_rows = extract_final_rows_from_csv(file_path)
                    mean_row = final_rows.iloc[0].values
                    std_row = final_rows.iloc[1].values
                    min_row = final_rows.iloc[2].values
                    max_row = final_rows.iloc[3].values

                    results.append({
                        'folder': folder_name,
                        'lr': lr,
                        'reg': reg,
                        'sigma': sigma,

                        # dice
                        'mean_mean_dice': mean_row[4],
                        # 'mean_std_dice': std_row[3],
                        # 'mean_min_dice': min_row[3],
                        # 'mean_max_dice': max_row[3],

                        # frac_foldings
                        'mean_frac_foldings': mean_row[5],
                        # 'mean_std_frac_foldings': std_row[4],
                        # 'mean_min_frac_foldings': min_row[4],
                        # 'mean_max_frac_foldings': max_row[4],

                        # sdlogj
                        'mean_mean_sdlogj': mean_row[-1],
                        # 'mean_std_sdlogj': std_row[-1],
                        # 'mean_min_sdlogj': min_row[-1],
                        # 'mean_max_sdlogj': max_row[-1]
                    })

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    return results_df


# Define the base directory
base_directory = '/u/home/koeglf/Documents/code/registrationbaselines/tmp/results_paramsearch'

# Run the main function and display the results
results_df = main(base_directory)

# Display the DataFrame
print(results_df)

# Analyze to find the best hyperparameters
# Example: Find the best based on highest mean_dice
best_hyperparameters_for_mean = results_df.loc[results_df['mean_mean_dice'].idxmax(
)]
best_hyperparameters_for_frac_foldings = results_df.loc[results_df['mean_frac_foldings'].idxmin(
)]
best_hyperparameters_for_sdlogj = results_df.loc[results_df['mean_mean_sdlogj'].idxmin(
)]

print("\n\n")
print("\n\n")
print("\n\n")

print("Best mean_dice:")
print(best_hyperparameters_for_mean.loc['folder'])
print("mean:\t\t", best_hyperparameters_for_mean.loc['mean_mean_dice'])
print("frac_foldings\t",
      best_hyperparameters_for_mean.loc['mean_frac_foldings'])
print("sdlogj:\t\t", best_hyperparameters_for_mean.loc['mean_mean_sdlogj'])
print("\n\n")

print("Best frac_foldings:")
print(best_hyperparameters_for_frac_foldings.loc['folder'])
print("mean:\t\t",
      best_hyperparameters_for_frac_foldings.loc['mean_mean_dice'])
print("frac_foldings\t",
      best_hyperparameters_for_frac_foldings.loc['mean_frac_foldings'])
print("sdlogj:\t\t",
      best_hyperparameters_for_frac_foldings.loc['mean_mean_sdlogj'])
print("\n\n")

print("Best sdlogj:")
print(best_hyperparameters_for_sdlogj.loc['folder'])
print("mean:\t\t", best_hyperparameters_for_sdlogj.loc['mean_mean_dice'])
print("frac_foldings\t",
      best_hyperparameters_for_sdlogj.loc['mean_frac_foldings'])
print("sdlogj:\t\t", best_hyperparameters_for_sdlogj.loc['mean_mean_sdlogj'])

# Scatter plot and correlation analysis


def plot_and_corr(x, y, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Scatter Plot: {xlabel} vs {ylabel}', fontsize=14)
    plt.show()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        correlation, _ = pearsonr(x, y)
        print(
            f'Pearson correlation between {xlabel} and {ylabel}: {correlation:.3f}')

        plt.savefig(
            f'/u/home/koeglf/Documents/code/registrationbaselines/examples/analysos/{xlabel}_vs_{ylabel}.png')


# Learning rate vs mean_mean_dice
plot_and_corr(np.array([a[0][0] for a in results_df['sigma']]), results_df['mean_mean_dice'],
              'sigma', 'Mean Dice Score')

# Learning rate vs mean_frac_foldings
plot_and_corr(np.array([a[0][0] for a in results_df['sigma']]), results_df['mean_frac_foldings'],
              'sigma', 'Mean Frac Foldings')

# Learning rate vs mean_mean_sdlogj
plot_and_corr(np.array([a[0][0] for a in results_df['sigma']]), results_df['mean_mean_sdlogj'],
              'sigma', 'Mean SDLogJ')


# Learning rate vs mean_mean_dice
plot_and_corr(np.array([a[0] for a in results_df['reg']]), results_df['mean_mean_dice'],
              'regularisation', 'Mean Dice Score')

# Learning rate vs mean_frac_foldings
plot_and_corr(np.array([a[0] for a in results_df['reg']]), results_df['mean_frac_foldings'],
              'regularisation', 'Mean Frac Foldings')

# Learning rate vs mean_mean_sdlogj
plot_and_corr(np.array([a[0] for a in results_df['reg']]), results_df['mean_mean_sdlogj'],
              'regularisation', 'Mean SDLogJ')
