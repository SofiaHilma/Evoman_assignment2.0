from matplotlib import rcParams
rcParams['font.family'] = "serif"     
rcParams['font.size']=17
from evoman.environment import Environment
from demo_controller import player_controller
from matplotlib import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

enemy_list = [1,2,3]
n_neuron = 10
    

elitist_alg_best_ind = [] # three best individuals when run against enemy 1,2,3 respectively
diverse_alg_best_ind = [] # three best individuals when run against enemy 1,2,3 respectively

scores_of_EliAlg_run_against_en1 = [] # There will be 5+5+5 scores in each list
scores_of_EliAlg_run_against_en2 = [] 
scores_of_EliAlg_run_against_en3 = []
scores_of_DivAlg_run_against_en1 = []
scores_of_DivAlg_run_against_en2 = []
scores_of_DivAlg_run_against_en3 = []

path1 = 'experiments/enemy_1'
path2 = 'experiments/enemy_2'
path3 = 'experiments/enemy_3'
# Extract the best individuals from the csv files
# Elitist
elitist_best = pd.read_csv(f'{path1}/elitist_best_enemy1.csv') # Elitist - enemy1
eli_ordered_best = elitist_best.sort_values(by='Fitness', ascending=False)
eli_best_string = eli_ordered_best['Genome'][0]
eli_best_stripped = eli_best_string.replace('[', '').replace(']', '')
eli_best_genome_enemy1 = [float(x) for x in eli_best_stripped.split()]
elitist_alg_best_ind.append(eli_best_genome_enemy1) 

elitist_best = pd.read_csv(f'{path2}/elitist_best_enemy2.csv') # Elitist - enemy2
eli_ordered_best = elitist_best.sort_values(by='Fitness', ascending=False)
eli_best_string = eli_ordered_best['Genome'][0]
eli_best_stripped = eli_best_string.replace('[', '').replace(']', '')
eli_best_genome_enemy2 = [float(x) for x in eli_best_stripped.split()]
elitist_alg_best_ind.append(eli_best_genome_enemy2) 

elitist_best = pd.read_csv(f'{path3}/elitist_best_enemy3.csv') # Elitist - enemy3
eli_ordered_best = elitist_best.sort_values(by='Fitness', ascending=False)
eli_best_string = eli_ordered_best['Genome'][0]
eli_best_stripped = eli_best_string.replace('[', '').replace(']', '')
eli_best_genome_enemy3 = [float(x) for x in eli_best_stripped.split()]
elitist_alg_best_ind.append(eli_best_genome_enemy3) 

# Diverse
diverse_best = pd.read_csv(f'{path1}/diverse_best_enemy1.csv') # Diverse - enemy1
div_ordered_best = diverse_best.sort_values(by='Fitness', ascending=False)
div_best_string = div_ordered_best['Genome'][0]
div_best_stripped = div_best_string.replace('[', '').replace(']', '')
div_best_genome_enemy1 = [float(x) for x in div_best_stripped.split()]
diverse_alg_best_ind.append(div_best_genome_enemy1) 


diverse_best = pd.read_csv(f'{path2}/diverse_best_enemy2.csv') # Diverse - enemy2
div_ordered_best = diverse_best.sort_values(by='Fitness', ascending=False)
div_best_string = div_ordered_best['Genome'][0]
div_best_stripped = div_best_string.replace('[', '').replace(']', '')
div_best_genome_enemy2 = [float(x) for x in div_best_stripped.split()]
diverse_alg_best_ind.append(div_best_genome_enemy2) 


diverse_best = pd.read_csv(f'{path3}/diverse_best_enemy3.csv') # Diverse - enemy3
div_ordered_best = diverse_best.sort_values(by='Fitness', ascending=False)
div_best_string = div_ordered_best['Genome'][0]
div_best_stripped = div_best_string.replace('[', '').replace(']', '')
div_best_genome_enemy3 = [float(x) for x in div_best_stripped.split()]
diverse_alg_best_ind.append(div_best_genome_enemy3)



# Pairs of enemies to test each individual against
enemy_pairs = [(1, 2), (0, 2), (0, 1)]


for i in range(3): # Go through the 6 individuals, but they are in 2 lists of three
    k, l = enemy_pairs[i]  # Select the enemy pairs to test against (the ones it was not run against before)

    for j in range(5): # -> we'll have 6 lists of 10 scores
        env = Environment(enemies=[i+1], logs="off", savelogs="no")
    
        # Test Elitist individual i against enemy k and l
        best_elitist_individual = np.array(elitist_alg_best_ind[i])
        best_diverse_individual = np.array(diverse_alg_best_ind[i])

        _, player_energy_against_k_eli, enemy_energy_k_eli, _ = env.play(best_elitist_individual, enemy_list[0]) # Test 'elitist alg run against i' against enemy k
        score_against_k_eli = player_energy_against_k_eli - enemy_energy_k_eli
        print(f"Scores (Elitist against enemy {k}): {score_against_k_eli}")

        _, player_energy_against_l_eli, enemy_energy_l_eli, _ = env.play(best_elitist_individual, enemy_list[1]) # Test 'elitist alg run against i' against enemy l
        score_against_l_eli = player_energy_against_l_eli - enemy_energy_l_eli       

        _, player_energy_against_l_eli, enemy_energy_l_eli, _ = env.play(best_elitist_individual, enemy_list[2]) 
        score_against_3_eli = player_energy_against_l_eli - enemy_energy_l_eli       

        _, player_energy_against_k_div, enemy_energy_k_div, _ = env.play(best_diverse_individual, enemy_list[0]) # Test 'diverse alg run against i' against enemy k
        score_against_k_div = player_energy_against_k_div - enemy_energy_k_div

        _, player_energy_against_l_div, enemy_energy_l_div, _ = env.play(best_diverse_individual, enemy_list[1]) # Test 'diverse alg run against i' against enemy l
        score_against_l_div = player_energy_against_l_div - enemy_energy_l_div      

        _, player_energy_against_k_div, enemy_energy_k_div, _ = env.play(best_diverse_individual, enemy_list[2]) 
        score_against_3_div = player_energy_against_k_div - enemy_energy_k_div

        # Append results to the lists
        if i == 0:
            scores_of_EliAlg_run_against_en1.append(score_against_k_eli)
            scores_of_EliAlg_run_against_en1.append(score_against_l_eli)
            scores_of_EliAlg_run_against_en1.append(score_against_3_eli)
            scores_of_DivAlg_run_against_en1.append(score_against_k_div)
            scores_of_DivAlg_run_against_en1.append(score_against_l_div)
            scores_of_DivAlg_run_against_en1.append(score_against_3_div)
        elif i == 1:
            scores_of_EliAlg_run_against_en2.append(score_against_k_eli)
            scores_of_EliAlg_run_against_en2.append(score_against_l_eli)
            scores_of_EliAlg_run_against_en1.append(score_against_3_eli)
            scores_of_DivAlg_run_against_en2.append(score_against_k_div)
            scores_of_DivAlg_run_against_en2.append(score_against_l_div)
            scores_of_DivAlg_run_against_en1.append(score_against_3_div)
        elif i == 2:
            scores_of_EliAlg_run_against_en3.append(score_against_k_eli)
            scores_of_EliAlg_run_against_en3.append(score_against_l_eli)
            scores_of_EliAlg_run_against_en1.append(score_against_3_eli)
            scores_of_DivAlg_run_against_en3.append(score_against_k_div)
            scores_of_DivAlg_run_against_en3.append(score_against_l_div)
            scores_of_DivAlg_run_against_en1.append(score_against_3_div)




        # find the average of each list? Not sure if it's necessary for boxplot

# Boxplot below

data = [scores_of_EliAlg_run_against_en1, scores_of_DivAlg_run_against_en1, scores_of_EliAlg_run_against_en2, scores_of_DivAlg_run_against_en2, scores_of_EliAlg_run_against_en3, scores_of_DivAlg_run_against_en3]

# Create boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(data, labels=[
    'El x 1', 'Div x 1',
    'El x 2', 'Div x 2',
    'El x 3', 'Div x 3'
])
plt.title('Averaged gain of best performing genomes')
plt.ylabel('Individual gain')
# plt.grid(True)
plt.show()

from scipy import stats
# Perform statistical tests
# Example: T-tests comparing each pair of algorithms for each enemy
for i in range(3):
    elitist_data = scores_of_EliAlg_run_against_en1  
    diverse_data = scores_of_DivAlg_run_against_en1
    
    # T-test (use appropriate test depending on your data distribution)
    t_stat, p_value = stats.ttest_ind(elitist_data, diverse_data)

    print(f"Enemy {i + 1} - T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    # Check if p-value is significant at 0.05 level
    if p_value < 0.05:
        print(f"Significant difference between Elitist and Diverse for Enemy {i + 1}.")
    else:
        print(f"No significant difference between Elitist and Diverse for Enemy {i+1}.")