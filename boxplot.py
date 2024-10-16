import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from evoman.environment import Environment
from demo_controller import player_controller

# Step 1: Load Best Individuals from Multiple CSV Files
def load_genomes_from_directory(directory, string, nostring):
    """Load all best individuals from CSV files in the directory."""
    genomes = []
    for filename in os.listdir(directory):
        if string in filename and nostring not in filename:
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            genome_string = df['Genome'].values[0]
            genome_string_cleaned = genome_string.replace('[', '').replace(']', '').strip()

            # Split the cleaned string by spaces and convert to a list of floats
            genome = [float(num) for num in genome_string_cleaned.split()]

            # Append the genome list to the result
            genomes.append(genome)
    return genomes
# Directory paths for the saved best individuals
triggered_diverse_dir = "C:/Users/anube/evoman_framework/experiments/enemy_group_1_2_3"
diverse_dir = "C:/Users/anube/evoman_framework/experiments/enemy_group_1_2_3"

genomes_triggered = np.array(load_genomes_from_directory(triggered_diverse_dir, "triggered", "gnome"))
genomes_diverse = np.array(load_genomes_from_directory(diverse_dir, "diverse", "triggered"))

# Define the enemy groups
enemy_group1 = [1, 2, 3]
enemy_group2 = [4, 5, 6]


# This function will run an individual (genome) against all enemies and calculate the gain
def evaluate_individual_against_enemies(genome, enemies):
    """Simulate the individual with the given genome against all enemies and return gains."""
    env = Environment(enemies=enemies, multiplemode="yes", player_controller=player_controller(10), logs = "off", savelogs="no")
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
triggered_gains_group1 = evaluate_all_genomes(genomes_triggered, enemy_group1)
triggered_gains_group2 = evaluate_all_genomes(genomes_triggered, enemy_group2)
diverse_gains_group1 = evaluate_all_genomes(genomes_diverse, enemy_group1)
diverse_gains_group2 = evaluate_all_genomes(genomes_diverse, enemy_group2)


def plot_boxplot(data, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=["Algorithm 1", "Algorithm 2"])
    plt.title(title)
    plt.ylabel("Gain (Player Health - Enemy Health)")
    plt.grid(True)
    plt.show()


# Plot for both enemy groups
plot_boxplot([triggered_gains_group1, diverse_gains_group1], "Gain Comparison for Enemy Group 1")
plot_boxplot([triggered_gains_group2, diverse_gains_group2], "Gain Comparison for Enemy Group 2")
