import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_paths_and_labels():
    # List of file paths
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
        r'results_set2/BSplines_feat_DINOv2_1.0_COSINE__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_l1 = base_dir + \
        r'results_set2/BSplines_feat_DINOv2_1.0_L1__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]_dino_upsample14/results.csv'

    #
    file_paths = [path_zero_disp,
                  path_ncc,
                  path_medsam_cosine,
                  path_medsam_l1,
                  path_sam_cosine,
                  path_sam_l1,
                  path_dinov2_cosine,
                  path_dinov2_l1]
    names = ["Initial",
             "NCC(image)",
             "cosine(MedSAM)",
             "L1(MedSAM)",
             "cosine(SAM)",
             "L1(SAM)",
             "cosine(DINOv2)",
             "L1(DINOv2)"]

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
    # Define a custom palette
    gray = ["#d1d1d1"]
    yellow = ["#fffd8a"]
    blues = ["#aed6e5", "#6cb4d0", "#51a6c8"]
    oranges = ["#ffc370", '#ffac38', '#e17e2d']
    purples = ["#d9b8ff", "#c08aff", "#a04dff"]
    custom_palette = gray + yellow + blues[:2] + oranges[:2] + purples[:2]

    # Plotting function
    plt.figure(figsize=(figure_width, 6))
    plt.rcParams['axes.linewidth'] = line_width*0.4
    box_line_fac = 0.4
    g = sns.catplot(data=df_melted,
                    width=0.7,
                    x="Metric",
                    y="Value",
                    hue="Source",
                    kind="box",
                    palette=custom_palette,  # Use the custom palette
                    legend='full',
                    boxprops={"edgecolor": "black",
                              "linewidth": line_width*box_line_fac},
                    whiskerprops={"color": "black",
                                  "linewidth": line_width*box_line_fac},
                    capprops={"color": "black",
                              "linewidth": line_width*box_line_fac},
                    medianprops={"color": "#b00202", "linestyle": "-",
                                 "linewidth": line_width*box_line_fac * 1.3})
    g.legend.set_bbox_to_anchor((0.3, 0.24))
    g.legend.set_title('')
    plt.setp(g._legend.get_texts(), fontsize=6)

    plt.ylim(0.15, 1.0)

    sns.despine(right=False,
                left=False,
                top=False,
                bottom=False)

    # set tick width
    plt.tick_params(axis='both', which='both',
                    width=line_width*0.4, direction='in')

    plt.xlabel('')
    plt.ylabel('DICE score')

    plt.tight_layout()
    plt.show()
    plt.savefig(
        'registrationbaselines/examples/visualisation_scripts/dice_boxplots/boxplots.png',
        dpi=300)

    x = 0


# Plot the boxplots
lim = -1
file_paths, names = get_paths_and_labels()
plot_boxplots(file_paths, names)
