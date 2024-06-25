import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_paths_and_labels():
    # List of file paths
    base_dir = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/'

    path_dinov2_00_cosine_10_ncc = base_dir + \
        r'results_set2/BSplines_no_encoder_1.0_NCC__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'
    path_dinov2_01_cosine_09_ncc = base_dir + \
        r'results_set4/BSplines_feat_DINOv2_0.1_COSINE_0.9_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_02_cosine_08_ncc = base_dir + \
        r'results_set4/BSplines_feat_DINOv2_0.2_COSINE_0.8_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_03_cosine_07_ncc = base_dir + \
        r'results_set4/BSplines_feat_DINOv2_0.3_COSINE_0.7_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    # path_dinov2_04_cosine_06_ncc = base_dir + \
    #     r'results_set4/BSplines_feat_DINOv2_0.4_COSINE_0.6_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_05_cosine_05_ncc = base_dir + \
        r'results_set3/BSplines_feat_DINOv2_0.5_COSINE_0.5_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    # path_dinov2_06_cosine_04_ncc = base_dir + \
    #     r'results_set4/BSplines_feat_DINOv2_0.6_COSINE_0.4_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    # path_dinov2_07_cosine_03_ncc = base_dir + \
    #     r'results_set4/BSplines_feat_DINOv2_0.7_COSINE_0.3_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    # path_dinov2_08_cosine_02_ncc = base_dir + \
    #     r'results_set4/BSplines_feat_DINOv2_0.8_COSINE_0.2_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    # path_dinov2_09_cosine_01_ncc = base_dir + \
    #     r'results_set4/BSplines_feat_DINOv2_0.9_COSINE_0.1_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_10_cosine_00_ncc = base_dir + \
        r'results_set2/BSplines_feat_DINOv2_1.0_COSINE__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]_dino_upsample14/results.csv'

    #
    file_paths = [path_dinov2_00_cosine_10_ncc,
                  path_dinov2_01_cosine_09_ncc,
                  path_dinov2_02_cosine_08_ncc,
                  path_dinov2_03_cosine_07_ncc,
                  # path_dinov2_04_cosine_06_ncc,
                  path_dinov2_05_cosine_05_ncc,
                  # path_dinov2_06_cosine_04_ncc,
                  # path_dinov2_07_cosine_03_ncc,
                  # path_dinov2_08_cosine_02_ncc,
                  # path_dinov2_09_cosine_01_ncc,
                  path_dinov2_10_cosine_00_ncc]

    names = ["0.0*cosine(DINOv2) + 1.0*NCC(image)",
             "0.1*cosine(DINOv2) + 0.9*NCC(image)",
             "0.2*cosine(DINOv2) + 0.8*NCC(image)",
             "0.3*cosine(DINOv2) + 0.7*NCC(image)",
             # "0.4*cosine(DINOv2) + 0.6*NCC(image)",
             "0.5*cosine(DINOv2) + 0.5*NCC(image)",
             # "0.6*cosine(DINOv2) + 0.4*NCC(image)",
             # "0.7*cosine(DINOv2) + 0.3*NCC(image)",
             # "0.8*cosine(DINOv2) + 0.2*NCC(image)",
             # "0.9*cosine(DINOv2) + 0.1*NCC(image)",
             "1.0*cosine(DINOv2) + 0.0*NCC(image)"]

    return file_paths, names


def load_and_process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    # Remove the last four rows
    df = df[:-4]
    # Extract the required columns
    df_extracted = df[['dice_0', 'dice_1', 'dice_2', 'dice_mean']]

    # rename to RV, LV-Mayo, LV, Mean
    df_extracted = df_extracted.rename(columns={'dice_0': 'RV',
                                                'dice_1': 'LV-Mayo',
                                                'dice_2': 'LV',
                                                'dice_mean': 'Mean'})

    return df_extracted


def plot_boxplots(file_paths, custom_labels, line_width=2, palette='Set3'):
    if len(file_paths) != len(custom_labels):
        raise ValueError(
            "The length of file_paths and custom_labels must be the same")

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through each file path and process
    for file_path, label in zip(file_paths, custom_labels):
        df_extracted = load_and_process_csv(file_path)
        df_extracted['Source'] = label  # Use custom label instead of file path
        dataframes.append(df_extracted)

    # Concatenate all dataframes
    df_concatenated = pd.concat(dataframes)

    # Melt the concatenated dataframe
    df_melted = df_concatenated.melt(
        id_vars='Source', var_name='Metric', value_name='Value')

    # Determine the figure width based on the number of files
    num_files = len(file_paths)
    figure_width = 9 + 2 * (num_files - 1)

    # Plotting function
plt.figure(figsize=(figure_width, 6))
plt.rcParams['axes.linewidth'] = line_width*0.4
sns.lineplot(data=df_melted[df_melted['Metric'] ==
             'Mean'], x="Source", y="Value", ci='sd')
sns.despine(right=False,
            left=False,
            top=False,
            bottom=False)
plt.xlabel('')
plt.ylabel('Mean Dice Score')

plt.tight_layout()
plt.show()
plt.savefig(
    'registrationbaselines/examples/visualisation_scripts/alpha_line_plot/line_plot.png',
    dpi=300,
    bbox_inches='tight', pad_inches=0.01)

x = 0


# Plot the boxplots
lim = -1
file_paths, names = get_paths_and_labels()
plot_boxplots(file_paths, names)
