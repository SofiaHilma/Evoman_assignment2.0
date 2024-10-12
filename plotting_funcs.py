import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "serif"     
rcParams['font.size']=17
from scipy.spatial.distance import pdist, squareform

'''def plot_avg_fitness(enemygroup, folder=None, ax=None, legend=True):
        Creates plot of averaged fitness per algorithm as well as best performing
     individual at every generation. 
     the input for enemy will be enemy_group_str which is for ex 1_2_3
    
    # Read triggered diverse data
    triggered_diverse_fitness = pd.read_csv(f'{folder}/enemy_group_{enemygroup}/triggered_diverse_fitness{enemygroup}.csv')
    print(triggered_diverse_fitness.head())
    mean_triggered_diverse = triggered_diverse_fitness.groupby('Generation')['Mean Fitness'].mean()
    std_triggered_diverse = triggered_diverse_fitness.groupby('Generation')['Std Fitness'].mean()
    best_triggered_diverse = triggered_diverse_fitness.groupby('Generation')['Best Fitness'].max()

    # Read diverse data
    diverse_fitness = pd.read_csv(f'{folder}/enemy_group_{enemygroup}/diverse_fitness{enemygroup}.csv')
    mean_diverse = diverse_fitness.groupby('Generation')['Mean Fitness'].mean()
    std_diverse = diverse_fitness.groupby('Generation')['Std Fitness'].mean()
    best_diverse = diverse_fitness.groupby('Generation')['Best Fitness'].max()

    if ax is None:
        plt.figure(figsize=(10, 6), dpi=300)

        # Plot average triggered diverse + std
        plt.plot(mean_triggered_diverse.index, mean_triggered_diverse.values, label='Triggered Divers', color='#69515A')
        plt.fill_between(mean_triggered_diverse.index,
                        np.array(mean_triggered_diverse.values) - np.array(std_triggered_diverse.values),
                        np.array(mean_triggered_diverse.values) + np.array(std_triggered_diverse.values),
                        color='#69515A', alpha=0.2)
        plt.plot(mean_triggered_diverse.index, best_triggered_diverse, label='Best Triggered Diverse', color='#59935C')
        
        # Plot average diverse + std
        plt.plot(mean_diverse.index, mean_diverse.values, label='Diverse', color='#59935C')
        plt.fill_between(mean_diverse.index,
                        np.array(mean_diverse.values) - np.array(std_diverse.values),
                        np.array(mean_diverse.values) + np.array(std_diverse.values),
                        color='#59935C', alpha=0.2)
        plt.plot(mean_diverse.index, best_diverse, label='Best Diverse', color='darkolivegreen')

        plt.title(f'Average fitness for both algorithms against enemy group {enemygroup}')
        plt.xlabel('Generation number')
        plt.ylabel('Fitness')
        plt.grid(True)
        if legend:
            plt.legend()
        plt.show()
        plt.savefig(f'{folder}/enemy_group_{enemygroup}/fitness_plot_enemy{enemygroup}.png', bbox_inches='tight')
        plt.close()
    else:
        # Plot average Triggered Diverse + std
        ax.plot(mean_triggered_diverse.index, mean_triggered_diverse.values, label='Triggered Diverse', color='#69515A')
        ax.fill_between(mean_triggered_diverse.index,
                        np.array(mean_triggered_diverse.values) - np.array(std_triggered_diverse.values),
                        np.array(mean_triggered_diverse.values) + np.array(std_triggered_diverse.values),
                        color='#69515A', alpha=0.2)
        ax.plot(mean_triggered_diverse.index, best_triggered_diverse, label='Best Triggered Diverse', color='#59935C')
        
        # Plot average diverse + std
        ax.plot(mean_diverse.index, mean_diverse.values, label='Diverse', color='#59935C')
        ax.fill_between(mean_diverse.index,
                        np.array(mean_diverse.values) - np.array(std_diverse.values),
                        np.array(mean_diverse.values) + np.array(std_diverse.values),
                        color='#59935C', alpha=0.2)
        ax.plot(mean_diverse.index, best_diverse, label='Best Diverse', color='darkolivegreen')

        ax.set_title(f'Enemy group {enemygroup}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid(True)
        if legend:
            ax.legend()
    '''

def plot_avg_fitness(enemygroup, folder=None, ax=None, legend=True):
    '''Creates plot of averaged fitness per algorithm as well as best performing
     individual at every generation. 
     the input for enemy will be enemy_group_str which is for ex 1_2_3'''
    
    # Read triggered diverse data
    triggered_diverse_fitness = pd.read_csv(f'{folder}/enemy_group_{enemygroup}/triggered_diverse_fitness{enemygroup}.csv')
    print(triggered_diverse_fitness.head())
    
    # Ensure DataFrame is not empty
    if triggered_diverse_fitness.empty:
        print("Triggered diverse fitness data is empty!")
        return

    mean_triggered_diverse = triggered_diverse_fitness.groupby('Generation')['Mean Fitness'].mean()
    std_triggered_diverse = triggered_diverse_fitness.groupby('Generation')['Std Fitness'].mean()
    best_triggered_diverse = triggered_diverse_fitness.groupby('Generation')['Best Fitness'].max()

    # Read diverse data
    diverse_fitness = pd.read_csv(f'{folder}/enemy_group_{enemygroup}/diverse_fitness{enemygroup}.csv')
    
    # Ensure DataFrame is not empty
    if diverse_fitness.empty:
        print("Diverse fitness data is empty!")
        return

    mean_diverse = diverse_fitness.groupby('Generation')['Mean Fitness'].mean()
    std_diverse = diverse_fitness.groupby('Generation')['Std Fitness'].mean()
    best_diverse = diverse_fitness.groupby('Generation')['Best Fitness'].max()

    # Create a new figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot average triggered diverse + std
    ax.plot(mean_triggered_diverse.index, mean_triggered_diverse.values, label='Triggered Diverse', color='#69515A')
    ax.fill_between(mean_triggered_diverse.index,
                    mean_triggered_diverse.values - std_triggered_diverse.values,
                    mean_triggered_diverse.values + std_triggered_diverse.values,
                    color='#69515A', alpha=0.2)
    ax.plot(mean_triggered_diverse.index, best_triggered_diverse, label='Best Triggered Diverse', color='#59935C')
    
    # Plot average diverse + std
    ax.plot(mean_diverse.index, mean_diverse.values, label='Diverse', color='#59935C')
    ax.fill_between(mean_diverse.index,
                    mean_diverse.values - std_diverse.values,
                    mean_diverse.values + std_diverse.values,
                    color='#59935C', alpha=0.2)
    ax.plot(mean_diverse.index, best_diverse, label='Best Diverse', color='darkolivegreen')

    # Add titles and labels
    ax.set_title(f'Average fitness for both algorithms against enemy group {enemygroup}')
    ax.set_xlabel('Generation number')
    ax.set_ylabel('Fitness')
    ax.grid(True)
    
    # Add legend if requested
    if legend:
        ax.legend()
    
    # Save the plot before showing
    plt.savefig(f'{folder}/enemy_group_{enemygroup}/fitness_plot_enemy{enemygroup}.png', bbox_inches='tight')
    
    # Show the plot
    plt.show()

    # Close the plot to free memory
    plt.close()


def plot_avg_diversity(enemygroup, folder=None, ax=None, legend=True):
    '''OLD: Creates plot of averaged diversity per algorithm, averaged over all specialists against all enemies.
    NEW: Creates a plot of diversity per algorithm, averaging a generalist (trained over a group)
    
    '''
    triggered_diverse = pd.DataFrame()
    diverse = pd.DataFrame()


    # Read diversity data
    
    triggered_diverse_enemygroup = pd.read_csv(f'{folder}/enemy_group_{enemygroup}/triggered_diverse_diversity{enemygroup}.csv')
    triggered_diverse = pd.concat([triggered_diverse, triggered_diverse_enemygroup])
    diverse_enemygroup = pd.read_csv(f'{folder}/enemy_group_{enemygroup}/diverse_diversity{enemygroup}.csv')
    diverse = pd.concat([diverse, diverse_enemygroup])

    # Average all data
    mean_triggered_diverse_diversity = triggered_diverse.groupby('generation')['mean_pairwise_distance'].mean()
    mean_diverse_diversity = diverse.groupby('generation')['mean_pairwise_distance'].mean()

     # Plot
    if ax is None:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(mean_triggered_diverse_diversity.index, mean_triggered_diverse_diversity.values, label='Triggered Diverse', color='indianred')
        plt.plot(mean_diverse_diversity.index, mean_diverse_diversity.values, label='Diverse', color='green')
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
        ax.plot(mean_triggered_diverse_diversity.index, mean_triggered_diverse_diversity.values, label='Triggered Diverse', color='indianred')
        ax.plot(mean_diverse_diversity.index, mean_diverse_diversity.values, label='Diverse', color='green')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Pairwise Distance')
        ax.set_title('Genetic Diversity')
        if legend:
            ax.legend()
        ax.grid(True)
    
    return ax


    '''
    # Read diversity data
    for enemy in enemies:
        enemy_elitist = pd.read_csv(f'{folder}/enemy_group_{enemy}/elitist_div_enemy{enemy}.csv')
        elitist = pd.concat([elitist,enemy_elitist])
        enemy_diverse = pd.read_csv(f'{folder}/enemy_{enemy}/diverse_div_enemy{enemy}.csv')
        diverse = pd.concat([diverse,enemy_diverse])
    '''
    '''    
    # Average all data
    mean_elitist_div = elitist.groupby('generation')['mean_pairwise_distance'].mean()
    mean_diverse_div = diverse.groupby('generation')['mean_pairwise_distance'].mean()

    '''

    '''# Plot
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
    '''