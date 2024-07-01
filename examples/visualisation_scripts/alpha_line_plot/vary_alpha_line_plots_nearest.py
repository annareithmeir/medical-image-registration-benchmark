import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_paths_and_labels():
    # List of file paths
    base_dir = r'/u/home/koeglf/Documents/code/registrationbaselines/tmp/nearest'

    path_dinov2_00_cosine_10_ncc = base_dir + \
        r'/BSplines_no_encoder_1.0_NCC__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]/results.csv'
    path_dinov2_01_cosine_09_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.1_COSINE_0.9_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_02_cosine_08_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.2_COSINE_0.8_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_03_cosine_07_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.3_COSINE_0.7_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_04_cosine_06_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.4_COSINE_0.6_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_05_cosine_05_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.5_COSINE_0.5_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_06_cosine_04_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.6_COSINE_0.4_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_07_cosine_03_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.7_COSINE_0.3_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_08_cosine_02_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.8_COSINE_0.2_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_09_cosine_01_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_0.9_COSINE_0.1_NCC__lr0.0005_reg[120]_it[1500]_sigma[[6, 6]]_dino_upsample14/results.csv'
    path_dinov2_10_cosine_00_ncc = base_dir + \
        r'/BSplines_feat_DINOv2_1.0_COSINE__lr0.0005_reg[120]_it[1000]_sigma[[6, 6]]_dino_upsample14/results.csv'

    #
    file_paths = [path_dinov2_00_cosine_10_ncc,
                  path_dinov2_01_cosine_09_ncc,
                  path_dinov2_02_cosine_08_ncc,
                  path_dinov2_03_cosine_07_ncc,
                  path_dinov2_04_cosine_06_ncc,
                  path_dinov2_05_cosine_05_ncc,
                  path_dinov2_06_cosine_04_ncc,
                  path_dinov2_07_cosine_03_ncc,
                  path_dinov2_08_cosine_02_ncc,
                  path_dinov2_09_cosine_01_ncc,
                  path_dinov2_10_cosine_00_ncc]

    # names = ["0.0⋅cosine(DINOv2)\n+ 1.0⋅NCC(image)",
    #          "0.1⋅cosine(DINOv2)\n+ 0.9⋅NCC(image)",
    #          "0.2⋅cosine(DINOv2)\n+ 0.8⋅NCC(image)",
    #          "0.3⋅cosine(DINOv2)\n+ 0.7⋅NCC(image)",
    #          # "0.4⋅cosine(DINOv2)\n+ 0.6⋅NCC(image)",
    #          "0.5⋅cosine(DINOv2)\n+ 0.5⋅NCC(image)",
    #          # "0.6⋅cosine(DINOv2)\n+ 0.4⋅NCC(image)",
    #          # "0.7⋅cosine(DINOv2)\n+ 0.3⋅NCC(image)",
    #          # "0.8⋅cosine(DINOv2)\n+ 0.2⋅NCC(image)",
    #          # "0.9⋅cosine(DINOv2)\n+ 0.1⋅NCC(image)",
    #          "1.0⋅cosine(DINOv2)\n+ 0.0⋅NCC(image)"]

    names = [0.0,
             0.1,
             0.2,
             0.3,
             0.4,
             0.5,
             0.6,
             0.7,
             0.8,
             0.9,
             1.0]

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


# Plot the boxplots
scaling_factor = 3
line_width = 2 * scaling_factor
palette = 'Set3'
file_paths, names = get_paths_and_labels()

if len(file_paths) != len(names):
    raise ValueError(
        "The length of file_paths and custom_labels must be the same")

# Initialize an empty list to store dataframes
dataframes = []

# Loop through each file path and process
for file_path, label in zip(file_paths, names):
    df_extracted = load_and_process_csv(file_path)
    df_extracted['Source'] = label  # Use custom label instead of file path
    dataframes.append(df_extracted)

# Concatenate all dataframes
df_concatenated = pd.concat(dataframes)

# Melt the concatenated dataframe
df_melted = df_concatenated.melt(
    id_vars='Source', var_name='Metric', value_name='Value')
df_melted = df_melted[df_melted['Metric'] == 'Mean']


# Calculate the mean and standard deviation for each 'Source'
df_stats = df_melted.groupby('Source')['Value'].agg(
    ['mean', 'std']).reset_index()
df_stats.columns = ['Source', 'Value', 'Std']

# Determine the figure width based on the number of files
num_files = len(file_paths)
figure_width = 2 * (num_files - 1)

plt.close()
fig, ax = plt.subplots(figsize=(figure_width, 6))
plt.grid(False)

plt.rcParams['axes.linewidth'] = line_width*0.4


plt.plot(df_stats['Source'], df_stats['Value'],
         color='black', linestyle='-', marker='o', linewidth=line_width, markersize=10, alpha=0.7)

# Adding error bars
plt.errorbar(df_stats['Source'], df_stats['Value'], yerr=df_stats['Std'],
             fmt='o', color='#a80000', capsize=12, alpha=0.7, linewidth=line_width, capthick=2*scaling_factor)

best_mean = df_stats['Value'].max()
best_source = df_stats[df_stats['Value'] == best_mean]['Source'].values[0]
print(f"Best mean: {best_mean} at {best_source}")

sns.despine(right=False,
            left=False,
            top=False,
            bottom=False)
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(line_width * 0.6)
ax.spines['bottom'].set_linewidth(line_width * 0.6)
ax.spines['left'].set_linewidth(line_width * 0.6)
ax.spines['right'].set_linewidth(line_width * 0.6)

# Turn off grid
plt.xlabel(r'$\alpha$', fontsize=16*scaling_factor)
# plt.xlabel(r'$\alpha$', fontsize=14*scaling_factor)
plt.ylabel('Mean DICE Score', fontsize=16*scaling_factor)
plt.xticks(fontsize=12*scaling_factor)
plt.yticks(fontsize=12*scaling_factor)

plt.tick_params(axis='both', which='both',
                width=line_width*0.6, direction='in', length=5)

plt.tight_layout()
plt.subplots_adjust(top=1.2)
plt.show()
plt.savefig(
    'registrationbaselines/examples/visualisation_scripts/alpha_line_plot/feature_alpha.png',
    dpi=300,
    bbox_inches='tight', pad_inches=0.01)
print("done")

x = 0
