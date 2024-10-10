# imports framework
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import pdist, squareform
rcParams['font.family'] = "serif"     
rcParams['font.size']=16

from algorithms import Algorithm_Elitist, Algorithm_Diverse
from plotting_funcs import plot_avg_fitness, plot_avg_diversity

# Select simulation settings
enemy_list = [1,2,3]
n_runs = 10
max_gens = 30

folder = 'test'
if not os.path.exists(folder):
    os.makedirs(folder)

# Run evolution experiments
# # for enemy in enemy_list:
#     enemy_folder = f'{folder}/enemy_{enemy}'
#     if not os.path.exists(enemy_folder):
#         os.makedirs(enemy_folder)

# Fitness data
elitist_enemy = pd.DataFrame() 
diverse_enemy = pd.DataFrame() 
# Diversity data
elitist_div_enemy = pd.DataFrame()
diverse_div_enemy = pd.DataFrame()
# Best individuals
elitist_best = pd.DataFrame()
diverse_best = pd.DataFrame()

for run in range(1, n_runs + 1):
    print(f'\nStarting Run {run} of {n_runs}...')

    # Initialize the algorithm
    elite_algorithm = Algorithm_Elitist(
        enemy=enemy_list,
        multiplemode='yes',
        n_neurons=10,
        low_weight=-1,
        upp_weight=1,
        pop_size=100,
        max_gens=max_gens
    )

    diverse_algorithm = Algorithm_Diverse(
        enemy=enemy_list,
        multiplemode='yes',
        low_weight=-1,
        upp_weight=1,
        pop_size=100,
        max_gens=max_gens
    )
    
    #Chose one of the EA to run or run both sequentialy
    print('Running Elitist Algorithm...')
    elite_algorithm.evolve(
        parent_selection=elite_algorithm.tournament_selection,
        crossover=elite_algorithm.crossover,
        mutation=elite_algorithm.uniform_mutation,
        survival_selection=elite_algorithm.elitist_selection,
        track_diversity=True
    )

    # Elitist Algorithm Results
    elitist_gen = pd.DataFrame({
        'Simulation': [run for _ in range(1,max_gens+1)],
        'Generation': range(1, len(elite_algorithm.best_fitness_over_time) + 1),
        'Best Fitness': elite_algorithm.best_fitness_over_time,
        'Mean Fitness': elite_algorithm.mean_fitness_over_time,
        'Std Fitness': elite_algorithm.std_fitness_over_time
    })

    elitist_enemy = pd.concat([elitist_enemy,elitist_gen])

    # Elitist Algorithm Diversity
    elitist_div_gen = pd.DataFrame(elite_algorithm.diversity_over_time)
    elitist_div_enemy = pd.concat([elitist_div_enemy, elitist_div_gen])

    # Elitist best individual tracking
    elitist_best_indiv = pd.DataFrame({
        'Genome': [elite_algorithm.best_individual[0]],
        'Fitness': [elite_algorithm.best_individual[1]]
    })
    elitist_best = pd.concat([elitist_best,elitist_best_indiv])


    print('Elitist Algorithm Completed.')

    print('Running Diverse Algorithm...')
    diverse_algorithm.evolve(
        parent_selection=diverse_algorithm.fitness_sharing,
        crossover=diverse_algorithm.crossover,
        mutation=diverse_algorithm.mutation,
        survival_selection=diverse_algorithm.roulette_wheel_selection,
        track_diversity=True
    )

    # Diverse Algorithm Results
    diverse_gen = pd.DataFrame({
        'Simulation': [run for _ in range(1,max_gens+1)],
        'Generation': range(1, len(diverse_algorithm.best_fitness_over_time) + 1),
        'Best Fitness': diverse_algorithm.best_fitness_over_time,
        'Mean Fitness': diverse_algorithm.mean_fitness_over_time,
        'Std Fitness': diverse_algorithm.std_fitness_over_time
    })
    diverse_enemy = pd.concat([diverse_enemy, diverse_gen])

    # Diverse Algorithm Diversity
    diverse_div_gen = pd.DataFrame(diverse_algorithm.diversity_over_time)
    diverse_div_enemy = pd.concat([diverse_div_enemy, diverse_div_gen])
    
    # Diverse best individual tracking
    diverse_best_indiv = pd.DataFrame({
        'Genome': [diverse_algorithm.best_individual[0]],
        'Fitness': [diverse_algorithm.best_individual[1]]
    })
    diverse_best = pd.concat([diverse_best,diverse_best_indiv])

    print('Diverse Algorithm Completed.')



# Save fitness data
elitist_enemy.to_csv(os.path.join(enemy_folder,f'elitist_enemy.csv'), index=False)
diverse_enemy.to_csv(os.path.join(enemy_folder,f'diverse_enemy.csv'), index=False)
# Save diversity data
elitist_div_enemy.to_csv(os.path.join(enemy_folder,f'elitist_div_enemy.csv'), index=False)
diverse_div_enemy.to_csv(os.path.join(enemy_folder,f'diverse_div_enemy.csv'), index=False)
# Save best individuals
elitist_best.to_csv(os.path.join(enemy_folder,f'elitist_best_enemy.csv'), index=False)
diverse_best.to_csv(os.path.join(enemy_folder,f'diverse_best_enemy.csv'), index=False)

# # Plot average fitness and diversity
# plot_avg_fitness(enemy, folder=folder)
# plot_avg_diversity([enemy], folder=folder)
# print(f'Plots for Enemy {enemy} generated succesfully.')

print('\nAll Experiments Completed.')




