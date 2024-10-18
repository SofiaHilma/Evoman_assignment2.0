import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller
from scipy.stats import ttest_ind, mannwhitneyu

group_1_dir = "C:/Users/anube/evoman_framework/experiments/enemy_group_1_2_3"
group_2_dir = "C:/Users/anube/evoman_framework/experiments/enemy_group_4_5_6"


def load_genomes_by_algorithm(directory, keyword_in, keyword_out):
    """Load all best individuals for a specific algorithm from CSV files in the directory."""
    genomes = []  # Reset the genomes list for each call
    for filename in os.listdir(directory):
        if keyword_out not in filename and keyword_in in filename:  # Look for the keyword (triggered/diverse) in the filename
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            # Load genomes row by row
            for index, row in df.iterrows():
                genome_string = row['Genome']
                genome_string_cleaned = genome_string.replace('[', '').replace(']', '').strip()

                # Split the cleaned string by spaces and convert to a list of floats
                genome = [float(num) for num in genome_string_cleaned.split()]
                genomes.append(genome)

            print(f"Loaded {len(genomes)} genomes from {filename}")

    return genomes

# Load genomes for each algorithm from both groups, ensuring no duplication
genomes_triggered_group1 = np.array(load_genomes_by_algorithm(group_1_dir, "triggered", "gnome"))
genomes_diverse_group1 = np.array(load_genomes_by_algorithm(group_1_dir, "diverse", "triggered"))

genomes_triggered_group2 = np.array(load_genomes_by_algorithm(group_2_dir, "triggered", "gnome"))
genomes_diverse_group2 = np.array(load_genomes_by_algorithm(group_2_dir, "diverse","triggered"))
# Define the enemy groups
enemies = [1, 2, 3, 4, 5, 6, 7, 8]



# This function will run an individual (genome) against all enemies and calculate the gain
def evaluate_individual_against_enemies(genome, enemies):
    """Simulate the individual with the given genome against all enemies and return gains."""
    env = Environment(enemies=enemies, multiplemode="yes", player_controller=player_controller(10), logs="off",
                      savelogs="no")
    fitness, player_life, enemy_life, time = env.play(pcont=genome)

    # Calculate gain as player health - enemy health
    total_gain = player_life - enemy_life

    return total_gain


# Evaluate all genomes from both algorithms
def evaluate_all_genomes(genomes, enemy_group):
    gains = []
    for genome in genomes:
        gain = evaluate_individual_against_enemies(genome, enemy_group)
        gains.append(gain)
    return gains


# Now, evaluate all individuals for each algorithm and enemy group
triggered_gains_group1 = evaluate_all_genomes(genomes_triggered_group1, enemies)
triggered_gains_group2 = evaluate_all_genomes(genomes_triggered_group2, enemies)
diverse_gains_group1 = evaluate_all_genomes(genomes_diverse_group1, enemies)
diverse_gains_group2 = evaluate_all_genomes(genomes_diverse_group2, enemies)

# Function to perform statistical testing
def perform_statistical_test(data1, data2):
    """Performs t-test and Mann-Whitney U test between two datasets."""
    t_stat, t_p_value = ttest_ind(data1, data2)
    mw_stat, mw_p_value = mannwhitneyu(data1, data2)
    return t_p_value, mw_p_value

combined_gains_group1 = triggered_gains_group1 + diverse_gains_group1
combined_gains_group2 = triggered_gains_group2 + diverse_gains_group2

# Perform statistical tests between the combined gains of Group 1 and Group 2
t_p_value_between_groups, mw_p_value_between_groups = perform_statistical_test(combined_gains_group1, combined_gains_group2)

print(f"T-test p-value between Group 1 and Group 2: {t_p_value_between_groups:.3e}")
print(f"Mann-Whitney U p-value between Group 1 and Group 2: {mw_p_value_between_groups:.3e}")

# Function to plot multiple boxplots with p-value annotations
def plot_combined_boxplots_with_pvalues(data_dict, title, t_p_value_between_groups, mw_p_value_between_groups):
    """Plots boxplots for diverse/triggered diverse algorithms with p-values annotations and between-group comparison."""
    num_plots = len(data_dict)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    # Iterate over the data and create a boxplot for each set
    for i, (subplot_title, data) in enumerate(data_dict.items()):
        ax = axes[i]
        ax.boxplot(data, labels=["Triggered Diverse", "Diverse"])
        ax.set_title(subplot_title, pad=30)
        ax.set_ylabel("Gain (Player Health - Enemy Health)")
        ax.grid(True)

        # Perform statistical test between the two algorithms within the same enemy group
        t_p_value, mw_p_value = perform_statistical_test(data[0], data[1])

        # Annotate p-values for each subplot
        ax.text(1.5, max([max(data[0]), max(data[1])]) + 2,
                f'T-test p-value: {t_p_value:.3e}\nMWU p-value: {mw_p_value:.3e}',
                horizontalalignment='center', fontsize=12, color='red')

    # Add a common p-value annotation for the between-group comparison (ignoring algorithms)
    middle_ax = axes[0]  # Reference the first plot to position the text between plots
    middle_ax.text(2.5, max([max(combined_gains_group1), max(combined_gains_group2)]) + 2,
                   f'Group Comparison\nT-test p-value: {t_p_value_between_groups:.3e}\nMWU p-value: {mw_p_value_between_groups:.3e}',
                   horizontalalignment='center', fontsize=12, color='blue')

    plt.tight_layout()
    plt.show()

# Boxplot data for the comparison between algorithms within groups
boxplot_data = {
    'Gain Comparison Group 1 (Enemies 1,2,3)': [triggered_gains_group1, diverse_gains_group1],
    'Gain Comparison Group 2 (Enemies 4,5,6)': [triggered_gains_group2, diverse_gains_group2]
}

# Call the plot function with group comparison p-values included
plot_combined_boxplots_with_pvalues(boxplot_data,
                                    "Gain Comparison with Statistical Testing",
                                    t_p_value_between_groups, mw_p_value_between_groups)
