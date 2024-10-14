import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "serif"     
rcParams['font.size']=17
from scipy.spatial.distance import pdist, squareform

# elitist_fitness = pd.read_csv(f'{folder}/enemy_{enemy}/elitist_enemy{enemy}.csv')
# diverse_fitness = pd.read_csv(f'{folder}/enemy_{enemy}/diverse_enemy{enemy}.csv')

def plot_avg_fitness(enemy,alg1,alg2, ax=None, legend=True):
    '''Creates plot of averaged fitness per algorithm as well as best performing
     individual at every generation. '''
    
    # Read alg1 data
    mean_alg1 =  alg1.groupby('Generation')['Mean Fitness'].mean()
    std_alg1 =   alg1.groupby('Generation')['Std Fitness'].mean()
    best_alg1 =  alg1.groupby('Generation')['Best Fitness'].max()

    # Read alg2 data
    mean_alg2 =  alg2.groupby('Generation')['Mean Fitness'].mean()
    std_alg2 =   alg2.groupby('Generation')['Std Fitness'].mean()
    best_alg2 =  alg2.groupby('Generation')['Best Fitness'].max()

    if ax is None:
        plt.figure(figsize=(10, 6), dpi=300)

        # Plot average alg1 + std
        plt.plot(mean_alg1.index, mean_alg1.values, label='alg1', color='#69515A')
        plt.fill_between(mean_alg1.index,
                        np.array(mean_alg1.values) - np.array(std_alg1.values),
                        np.array(mean_alg1.values) + np.array(std_alg1.values),
                        color='#69515A', alpha=0.2)
        plt.plot(mean_alg1.index, best_alg1, label='Best alg1', color='#59935C')
        
        # Plot average alg2 + std
        plt.plot(mean_alg2.index, mean_alg2.values, label='alg2', color='#59935C')
        plt.fill_between(mean_alg2.index,
                        np.array(mean_alg2.values) - np.array(std_alg2.values),
                        np.array(mean_alg2.values) + np.array(std_alg2.values),
                        color='#59935C', alpha=0.2)
        plt.plot(mean_alg1.index, best_alg2, label='Best alg2', color='darkolivegreen')

        plt.title(f'Average fitness for both algorithms against enemy {enemy}')
        plt.xlabel('Generation number')
        plt.ylabel('Fitness')
        plt.grid(True)
        if legend:
            plt.legend()
        plt.show()
        # plt.savefig(f'{folder}/enemy_{enemy}/fitness_plot_enemy{enemy}.png', bbox_inches='tight')
        plt.close()
    else:
        # Plot average alg1 + std
        ax.plot(mean_alg1.index, mean_alg1.values, label='alg1', color='#69515A')
        ax.fill_between(mean_alg1.index,
                        np.array(mean_alg1.values) - np.array(std_alg1.values),
                        np.array(mean_alg1.values) + np.array(std_alg1.values),
                        color='#69515A', alpha=0.2)
        ax.plot(mean_alg1.index, best_alg1, label='Best alg1', color='#59935C')
        
        # Plot average alg2 + std
        ax.plot(mean_alg2.index, mean_alg2.values, label='alg2', color='#59935C')
        ax.fill_between(mean_alg2.index,
                        np.array(mean_alg2.values) - np.array(std_alg2.values),
                        np.array(mean_alg2.values) + np.array(std_alg2.values),
                        color='#59935C', alpha=0.2)
        ax.plot(mean_alg1.index, best_alg2, label='Best alg2', color='darkolivegreen')

        ax.set_title(f'Enemy {enemy}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid(True)
        if legend:
            ax.legend()


def plot_avg_diversity(enemies, folder=None, ax=None, legend=True):
    '''Creates plot of averaged diversity per algorithm, averaged over all specialists against all enemies.'''
    alg1 = pd.DataFrame()
    alg2 = pd.DataFrame()
    
    # Read diversity data
    for enemy in enemies:
        enemy_alg1 = pd.read_csv(f'{folder}/alg1_div_enemy{enemy}.csv')
        alg1 = pd.concat([alg1,enemy_alg1])
        enemy_alg2 = pd.read_csv(f'{folder}/alg2_div_enemy{enemy}.csv')
        alg2 = pd.concat([alg2,enemy_alg2])

    # Average all data
    mean_alg1_div = alg1.groupby('generation')['mean_pairwise_distance'].mean()
    mean_alg2_div = alg2.groupby('generation')['mean_pairwise_distance'].mean()

    # Plot
    if ax is None:
        plt.figure(figsize=(10, 6), dpi=300)

        plt.plot(mean_alg1_div.index, mean_alg1_div.values, label='alg1', color='indianred')
        plt.plot(mean_alg2_div.index, mean_alg2_div.values, label='alg2', color='green')
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
        ax.plot(mean_alg1_div.index, mean_alg1_div.values, label='alg1', color='indianred')
        ax.plot(mean_alg2_div.index, mean_alg2_div.values, label='alg2', color='green')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Pairwise Distance')
        ax.set_title('Genetic Diversity')
        if legend:
            ax.legend()
        ax.grid(True)

        return ax