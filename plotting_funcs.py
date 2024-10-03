import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "serif"     
rcParams['font.size']=17
from scipy.spatial.distance import pdist, squareform

def plot_avg_fitness(enemy, folder=None, ax=None, legend=True):
    '''Creates plot of averaged fitness per algorithm as well as best performing
     individual at every generation. '''
    
    # Read elitist data
    elitist_fitness = pd.read_csv(f'{folder}/enemy_{enemy}/elitist_enemy{enemy}.csv')
    mean_elitist = elitist_fitness.groupby('Generation')['Mean Fitness'].mean()
    std_elitist = elitist_fitness.groupby('Generation')['Std Fitness'].mean()
    best_elitist = elitist_fitness.groupby('Generation')['Best Fitness'].max()

    # Read diverse data
    diverse_fitness = pd.read_csv(f'{folder}/enemy_{enemy}/diverse_enemy{enemy}.csv')
    mean_diverse = diverse_fitness.groupby('Generation')['Mean Fitness'].mean()
    std_diverse = diverse_fitness.groupby('Generation')['Std Fitness'].mean()
    best_diverse = diverse_fitness.groupby('Generation')['Best Fitness'].max()

    if ax is None:
        plt.figure(figsize=(10, 6), dpi=300)

        # Plot average elitist + std
        plt.plot(mean_elitist.index, mean_elitist.values, label='Elitist', color='#69515A')
        plt.fill_between(mean_elitist.index,
                        np.array(mean_elitist.values) - np.array(std_elitist.values),
                        np.array(mean_elitist.values) + np.array(std_elitist.values),
                        color='#69515A', alpha=0.2)
        plt.plot(mean_elitist.index, best_elitist, label='Best Elitist', color='#59935C')
        
        # Plot average diverse + std
        plt.plot(mean_diverse.index, mean_diverse.values, label='Diverse', color='#59935C')
        plt.fill_between(mean_diverse.index,
                        np.array(mean_diverse.values) - np.array(std_diverse.values),
                        np.array(mean_diverse.values) + np.array(std_diverse.values),
                        color='#59935C', alpha=0.2)
        plt.plot(mean_elitist.index, best_diverse, label='Best Diverse', color='darkolivegreen')

        plt.title(f'Average fitness for both algorithms against enemy {enemy}')
        plt.xlabel('Generation number')
        plt.ylabel('Fitness')
        plt.grid(True)
        if legend:
            plt.legend()
        plt.show()
        plt.savefig(f'{folder}/enemy_{enemy}/fitness_plot_enemy{enemy}.png', bbox_inches='tight')
        plt.close()
    else:
        # Plot average elitist + std
        ax.plot(mean_elitist.index, mean_elitist.values, label='Elitist', color='#69515A')
        ax.fill_between(mean_elitist.index,
                        np.array(mean_elitist.values) - np.array(std_elitist.values),
                        np.array(mean_elitist.values) + np.array(std_elitist.values),
                        color='#69515A', alpha=0.2)
        ax.plot(mean_elitist.index, best_elitist, label='Best Elitist', color='#59935C')
        
        # Plot average diverse + std
        ax.plot(mean_diverse.index, mean_diverse.values, label='Diverse', color='#59935C')
        ax.fill_between(mean_diverse.index,
                        np.array(mean_diverse.values) - np.array(std_diverse.values),
                        np.array(mean_diverse.values) + np.array(std_diverse.values),
                        color='#59935C', alpha=0.2)
        ax.plot(mean_elitist.index, best_diverse, label='Best Diverse', color='darkolivegreen')

        ax.set_title(f'Enemy {enemy}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid(True)
        if legend:
            ax.legend()


def plot_avg_diversity(enemies, folder=None, ax=None, legend=True):
    '''Creates plot of averaged diversity per algorithm, averaged over all specialists against all enemies.'''
    elitist = pd.DataFrame()
    diverse = pd.DataFrame()
    
    # Read diversity data
    for enemy in enemies:
        enemy_elitist = pd.read_csv(f'{folder}/enemy_{enemy}/elitist_div_enemy{enemy}.csv')
        elitist = pd.concat([elitist,enemy_elitist])
        enemy_diverse = pd.read_csv(f'{folder}/enemy_{enemy}/diverse_div_enemy{enemy}.csv')
        diverse = pd.concat([diverse,enemy_diverse])

    # Average all data
    mean_elitist_div = elitist.groupby('generation')['mean_pairwise_distance'].mean()
    mean_diverse_div = diverse.groupby('generation')['mean_pairwise_distance'].mean()

    # Plot
    if ax is None:
        plt.figure(figsize=(10, 6), dpi=300)

        plt.plot(mean_elitist_div.index, mean_elitist_div.values, label='Elitist', color='indianred')
        plt.plot(mean_diverse_div.index, mean_diverse_div.values, label='Diverse', color='green')
        plt.xlabel('Generation')
        plt.ylabel('Mean Pairwise Distance')
        plt.title('Genetic Diversity Over Generations')
        if legend:
            plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig(f'{folder}/diversity_plot.png', bbox_inches='tight')
        plt.close()
    else:
        ax.plot(mean_elitist_div.index, mean_elitist_div.values, label='Elitist', color='indianred')
        ax.plot(mean_diverse_div.index, mean_diverse_div.values, label='Diverse', color='green')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Pairwise Distance')
        ax.set_title('Genetic Diversity')
        if legend:
            ax.legend()
        ax.grid(True)

        return ax