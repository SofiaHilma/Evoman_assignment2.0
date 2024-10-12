# imports framework
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import pdist, squareform
rcParams['font.family'] = "serif"     
rcParams['font.size']=16

#from algorithms import Algorithm_Triggered_Diversity, Algorithm_Diverse
from algorithms import Evolution
from plotting_funcs import plot_avg_fitness, plot_avg_diversity

"""
PSEUDOCODE:
1. for loop over enemy groups (so two loops):
2.      make a folder for that enemy_group to store data results
3.          loop through the number of runs
4.              in a run, it runs the triggered diverse and the diverse algorithms and saves this data:
                    1. fitness data: best, mean, std
                    2. diversity data
                    3. best individual
                    4. 
"""

# Select simulation settings
enemy_group1 = [1,2,3]
enemy_group2 = [4,5,6] # now we'll run experiments not per enemy in list but per entire list (so two times)

n_runs = 5
max_gens = 10

folder = 'experiments'
if not os.path.exists(folder):
    os.makedirs(folder)

# Run evolution experiments per enemy group
for enemy_group in [enemy_group1, enemy_group2]:
    enemy_group_str = '_'.join(map(str, enemy_group))
    enemy_folder = f'{folder}/enemy_group_{enemy_group_str}'
    if not os.path.exists(enemy_folder):
        os.makedirs(enemy_folder)

    # Fitness data
    triggered_diverse_fitness = pd.DataFrame() # gone from elitist & diverse to triggered_diverse & diverse
    diverse_fitness = pd.DataFrame() 
    # Diversity data
    triggered_diverse_diversity = pd.DataFrame() 
    diverse_diversity = pd.DataFrame()
    # Best individuals
    triggered_diverse_best = pd.DataFrame()
    diverse_best = pd.DataFrame()

    for run in range(1, n_runs + 1):
        print(f'\nStarting Enemy Group {enemy_group}, Run {run} of {n_runs}...')

        # Initialize the algorithm
        triggered_diverse_algorithm = Evolution( 
            enemy=enemy_group,
            multiplemode="yes",
            n_neurons=10,
            low_weight=-1,
            upp_weight=1,
            pop_size=100,
            max_gens=max_gens
        )

        diverse_algorithm = Evolution(    
            enemy=enemy_group,
            multiplemode="yes",
            n_neurons=10,
            low_weight=-1,
            upp_weight=1,
            pop_size=100,
            max_gens=max_gens
        )
        
        # Chose one of the EA to run or run both sequentialy
        print('Running Triggered Diverse Algorithm...')
        triggered_diverse_algorithm.triggered_diverse_evolve(
            track_diversity=True
        )

        

        # Triggered Diverse Algorithm Results
        triggered_diverse_gen = pd.DataFrame({
            'Simulation': [run for _ in range(1,max_gens+1)],
            'Generation': range(1, len(triggered_diverse_algorithm.best_fitness_over_time) + 1),
            'Best Fitness': triggered_diverse_algorithm.best_fitness_over_time,
            'Mean Fitness': triggered_diverse_algorithm.mean_fitness_over_time,
            'Std Fitness': triggered_diverse_algorithm.std_fitness_over_time
        })

        triggered_diverse_fitness = pd.concat([triggered_diverse_fitness, triggered_diverse_gen])
 
        # Triggered Diverse Algorithm Diversity
        triggered_diverse_div_gen = pd.DataFrame(triggered_diverse_algorithm.diversity_over_time)
        triggered_diverse_diversity = pd.concat([triggered_diverse_diversity, triggered_diverse_div_gen])

        # Triggered Diverse best individual tracking
        triggered_diverse_best_indiv = pd.DataFrame({
            'Genome': [triggered_diverse_algorithm.best_individual[0]],
            'Fitness': [triggered_diverse_algorithm.best_individual[1]]
        })
        triggered_diverse_best = pd.concat([triggered_diverse_best,triggered_diverse_best_indiv])


        print('Triggered Diverse Algorithm Completed.')



        # Diverse Algorithm
        print('Running Diverse Algorithm...')
        diverse_algorithm.diverse_evolve(
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
        diverse_fitness = pd.concat([diverse_fitness, diverse_gen])

        # Diverse Algorithm Diversity
        diverse_div_gen = pd.DataFrame(diverse_algorithm.diversity_over_time)
        diverse_diversity = pd.concat([diverse_diversity, diverse_div_gen])
        
        # Diverse best individual tracking
        diverse_best_indiv = pd.DataFrame({
            'Genome': [diverse_algorithm.best_individual[0]],
            'Fitness': [diverse_algorithm.best_individual[1]]
        })
        diverse_best = pd.concat([diverse_best,diverse_best_indiv])

        print('Diverse Algorithm Completed.')



    # Save fitness data
    triggered_diverse_fitness.to_csv(os.path.join(enemy_folder,f'triggered_diverse_fitness{enemy_group_str}.csv'), index=False)
    diverse_fitness.to_csv(os.path.join(enemy_folder,f'diverse_fitness{enemy_group_str}.csv'), index=False)
    # Save diversity data
    triggered_diverse_diversity.to_csv(os.path.join(enemy_folder,f'triggered_diverse_diversity{enemy_group_str}.csv'), index=False)
    diverse_diversity.to_csv(os.path.join(enemy_folder,f'diverse_diversity{enemy_group_str}.csv'), index=False)
    # Save best individuals
    triggered_diverse_best.to_csv(os.path.join(enemy_folder,f'triggered_diverse_best_enemy{enemy_group_str}.csv'), index=False)
    diverse_best.to_csv(os.path.join(enemy_folder,f'diverse_best_enemy{enemy_group_str}.csv'), index=False)

    # Plot average fitness and diversity
    plot_avg_fitness(enemy_group_str, folder=folder)
    #plot_avg_diversity(enemy_group_str, folder=folder)
    print(f'Plots for Enemy {enemy_group_str} generated succesfully.')

print('\nAll Experiments Completed.')




