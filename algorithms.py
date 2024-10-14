# imports framework
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import pdist, squareform
rcParams['font.family'] = "serif"     
rcParams['font.size']=17
from evoman.environment import Environment
from demo_controller import player_controller


# We do one class for both algorithms now!!!!!
class Evolution:
    def __init__(self, enemy=[1], multiplemode = "no", n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        self.enemy = enemy
        self.n_neurons = n_neurons
        self.low_weight = low_weight
        self.upp_weight = upp_weight
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.env = Environment(enemies=enemy, multiplemode = "no", logs="off", savelogs="no", player_controller=player_controller(n_neurons))
        # Initialize population
        self.population = self.create_population()
        self.best_individual = None

        # Initialize tracking variables
        self.best_fitness_over_time = []
        self.mean_fitness_over_time = []
        self.std_fitness_over_time = []
        self.diversity_over_time = []

    def create_population(self):
        '''Creates a population of individuals with random weights within the range [low_weight, upp_weight]'''
        # Calculate the correct number of weights based on neural network structure
        n_inputs = self.env.get_num_sensors()  # Number of input neurons
        n_outputs = 5  # Number of output neurons (actions)

        # Calculate number of weights
        input_to_hidden_weights = (n_inputs + 1) * self.n_neurons  # +1 for bias
        hidden_to_output_weights = (self.n_neurons + 1) * n_outputs  # +1 for bias
        self.n_vars = input_to_hidden_weights + hidden_to_output_weights

        # Create population with correct number of weights
        return np.random.uniform(self.low_weight, self.upp_weight, (self.pop_size, self.n_vars))

    def simulate(self, population=None):
        '''Runs simulations with a given population or the current population by default'''
        if population is None:
            population = self.population  # Use the instance's population if no population is passed

        pop_fitness = []
        for individual in population:
            f, _, _, _ = self.env.play(pcont=individual)
            pop_fitness.append(f)

        return np.array(pop_fitness)
    
    
    
    '''def __init__(self, enemy=[1], multiplemode = "no", n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        super().__init__(enemy=enemy, multiplemode=multiplemode, n_neurons=n_neurons, low_weight=low_weight,
                         upp_weight=upp_weight, pop_size=pop_size, max_gens=max_gens)'''

    def roulette_wheel_selection(self, population, fitness, n_parents):
        # OLD ROULETTE WHEEL (NO SCALING I THINK)

        # Ensure the population is a NumPy array
        population = np.array(population)

        # Ensure fitness values are non-negative
        fitness = np.maximum(fitness, 0)  # This makes all negative fitness values zero

        # Calculate total fitness
        total_fitness = np.sum(fitness)

        # If total fitness is zero, use uniform random selection
        if total_fitness == 0:
            probabilities = np.ones(len(fitness)) / len(fitness)
        else:
            # Otherwise, calculate the probabilities proportional to fitness
            probabilities = fitness / total_fitness

        # Perform roulette wheel selection based on the calculated probabilities
        selected_indices = np.random.choice(len(population), size=n_parents, p=probabilities, replace=True)

        # Return the selected individuals, ensuring population is indexed as a NumPy array
        return population[selected_indices]
    
        # MY ROULETTE WHEEL
        """
        Roulette wheel selection (fitness-proportional selection).
        Selects individuals based on their fitness proportionally.
        Returns the selected individuals in a list.
        """
        '''# Ensure the population is a NumPy array
        population = np.array(population)

        # Normalise fitness values
        mean_fitness = np.mean(fitness) # Calculate mean and standard deviation of fitnesses
        std_fitness = np.std(fitness)
        if std_fitness > 0:
            normalised_fitness = (fitness - mean_fitness) / std_fitness
        else:
            normalised_fitness = np.zeros(len(fitness))  # If std is zero, reset all fitnesses to zero

        # Ensure normalised fitness values are non-negative
        fitness = np.maximum(normalised_fitness, 0)

        # Calculate probabilities for each individual to be selected
        total_fitness = np.sum(fitness)
        if total_fitness == 0: 
            # If total fitness is zero use uniform random selection
            probabilities = np.ones(len(fitness)) / len(fitness)
        else:
            # Otherwise calculate the probabilities proportional to fitness
            probabilities = fitness / total_fitness

        # Perform roulette wheel selection based on the calculated probabilities
        selected_indices = np.random.choice(len(population), size=n_parents, p=probabilities, replace=True)

        return population[selected_indices]'''



    # Uniform crossover
    def basic_crossover(self, parents, n_offspring):
        """
        Each gene is randomly chosen from any parent.
        'parents' is a list of parent genomes.
        Returns n_offspring children.
        """
        num_parents = len(parents)
        genome_length = len(parents[0])

        # Create an empty array to store the offspring
        data_type = parents[0].dtype
        offspring = np.zeros((n_offspring, genome_length), dtype=data_type)

        # Loop over each offspring to generate one at a time
        for i in range(n_offspring):
            child = np.zeros(genome_length, dtype=data_type)

            # Loop over each gene in the genome
            for gene in range(genome_length):
                # Randomly select a parent
                parent_index = np.random.randint(num_parents)
                child[gene] = parents[parent_index][gene]
                
            # Add the child to the offspring
            offspring[i] = child
        return offspring
    
    
    # Uniform mutation 
    def diverse_mutation(self, individual, mutation_rate):
        """
        Each gene has a mutation_rate chance of being replaced with a new random value.
        """

        # Loop over each gene in the genome        
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                # Replace the gene with a new random value within the valid range
                individual[i] = np.random.uniform(self.low_weight, self.upp_weight)
        return individual
    

    # Triggered Diversity: doomsday methods below

    # selection, crossover and mutation methods should be the same as in diverse alg??
    # And the trigger and calling is in the evolve method


### Diverse algorithm
    #def __init__(self, enemy=[1], multiplemode = "no", n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        # Call the superclass __init__ method to initialize population and other attributes
     #   super().__init__(enemy=enemy, multiplemode=multiplemode, n_neurons=n_neurons, low_weight=low_weight,
      #                   upp_weight=upp_weight, pop_size=pop_size, max_gens=max_gens)
       # self.diversity_over_time = []

    def fitness_sharing(self, population, fitness, n_parents, sigma_share=0.3):
        n_parents = min(n_parents, len(population))
        shared_fitness = np.copy(fitness)
        for i in range(len(population)):
            niche_count = sum(max(0, 1 - (np.linalg.norm(population[i] - population[j]) / sigma_share)**2)
                            for j in range(len(population)))
            shared_fitness[i] = max(1e-10, shared_fitness[i] / max(1, niche_count))  # Ensure non-negative values

        total_fitness = np.sum(shared_fitness)
        probabilities = shared_fitness / total_fitness

        # Ensure there are no negative probabilities
        min_prob = 1e-10
        probabilities = np.maximum(probabilities, min_prob)
        probabilities /= np.sum(probabilities) # Normalize probabilities

        try:
            selected_indices = np.random.choice(len(population), size=n_parents, p=probabilities, replace=False)

        except ValueError:
            selected_indices = np.random.choice(len(population), size=n_parents, p=probabilities, replace=True)

        return population[selected_indices]

    '''def roulette_wheel_selection(self, population, fitness, n_parents):
        """
        Roulette wheel selection (fitness-proportional selection).
        Selects individuals based on their fitness proportionally.
        """
        # Ensure the population is a NumPy array
        population = np.array(population)

        # Ensure fitness values are non-negative
        fitness = np.maximum(fitness, 0)  # This makes all negative fitness values zero

        # Calculate total fitness
        total_fitness = np.sum(fitness)

        # If total fitness is zero, use uniform random selection
        if total_fitness == 0:
            probabilities = np.ones(len(fitness)) / len(fitness)
        else:
            # Otherwise, calculate the probabilities proportional to fitness
            probabilities = fitness / total_fitness

        # Perform roulette wheel selection based on the calculated probabilities
        selected_indices = np.random.choice(len(population), size=n_parents, p=probabilities, replace=True)

        # Return the selected individuals, ensuring population is indexed as a NumPy array
        return population[selected_indices]'''


    def diverse_crossover(self, parents, n_offspring):
        # Generate pool from parents
        gene_pool = [[] for _ in range(len(parents[0]))]
        for parent in parents:
            for gene in range(len(parent)):
                gene_pool[gene].append(parent[gene])

        offspring = [[] for _ in range(n_offspring)]

        # Create children from pool
        for child in range(n_offspring):
            for gene in range(len(parents[0])):
                offspring[child].append(np.random.choice(gene_pool[gene]))

        return np.array(offspring)

    def basic_mutation(self, individual, mutation_rate, scale=0.6):
        # Implement mutation with a Cauchy distribution
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] += np.random.standard_cauchy() * scale
                individual[i] = np.clip(individual[i], self.low_weight, self.upp_weight)
        return individual
    

    def triggered_diverse_evolve(self, n_parents=10, mutation_rate=0.2,
               track_diversity=True):
        """Performs evolutionary algorithm with optional diversity tracking."""
        if track_diversity:
            self.track_diversity(self.population, generation=0)
        
        counter = 0
        for generation in range(1, self.max_gens + 1):

            if counter == 2:
                # Trigger diversity
                # Step 1: Evaluate fitness of the current population
                fitness_values = self.simulate()  # Method to evaluate population (assumed in Evolution)

                # Step 2: Select parents using the specified selection method
                parents = self.fitness_sharing(self.population, fitness_values, n_parents=n_parents)

                # Step 2: Generate Offspring via Crossover
                offspring = self.diverse_crossover(parents, n_offspring=self.pop_size)

                # Step 3: Mutate the Offspring
                mutated_offspring = []
                for child in offspring:
                    mutated_child = self.diverse_mutation(child, mutation_rate=mutation_rate)  # Apply mutation
                    mutated_offspring.append(mutated_child)

                mutated_offspring = np.array(mutated_offspring)

                # Step 4: Evaluate fitness of offspring
                offspring_fitness = self.simulate(mutated_offspring)

                # Step 5: Combine current population and offspring for survival selection
                combined_population = np.vstack((self.population, mutated_offspring))
                combined_fitness = np.concatenate((fitness_values, offspring_fitness))

                # Step 6: Survivor Selection (use the specified method for replacement)
                self.population = self.roulette_wheel_selection(combined_population, combined_fitness, n_parents=self.pop_size)

                # Step 6: Track fitness metrics
                self.track_fitness(combined_fitness[:self.pop_size])

                # Optional: Track genetic diversity metrics
                if track_diversity:
                    self.track_diversity(self.population, generation)
                counter = 0

                print(f'Diversity was triggered at generation:{generation}')

            else: 
                # Step 1: Evaluate fitness of the current population
                fitness_values = self.simulate()  # Method to evaluate population (assumed in Evolution)

                # Step 2: Select parents using the specified selection method
                parents = self.roulette_wheel_selection(self.population, fitness_values, n_parents=n_parents)

                # Step 2: Generate Offspring via Crossover
                offspring = self.basic_crossover(parents, n_offspring=self.pop_size)

                # Step 3: Mutate the Offspring
                mutated_offspring = []
                for child in offspring:
                    mutated_child = self.basic_mutation(child, mutation_rate=mutation_rate)  # Apply mutation
                    mutated_offspring.append(mutated_child)

                mutated_offspring = np.array(mutated_offspring)

                # Step 4: Evaluate fitness of offspring
                offspring_fitness = self.simulate(mutated_offspring)

                # Step 5: Combine current population and offspring for survival selection
                combined_population = np.vstack((self.population, mutated_offspring))
                combined_fitness = np.concatenate((fitness_values, offspring_fitness))

                # Step 6: Survivor Selection (use the specified method for replacement)
                self.population = self.roulette_wheel_selection(combined_population, combined_fitness, n_parents=self.pop_size)

                # Step 6: Track fitness metrics
                self.track_fitness(combined_fitness[:self.pop_size])

                # Optional: Track genetic diversity metrics
                if track_diversity:
                    self.track_diversity(self.population, generation)

                # If the fitness hasn't improved more than?? and if the fitness is below 90 add to the counter
                if len(self.mean_fitness_over_time) > 1:  # Check if there are at least two elements
                    if abs(self.mean_fitness_over_time[-1] - self.mean_fitness_over_time[-2]) < 10 and self.mean_fitness_over_time[-1]<90:
                        counter += 1 
                    
        best_index = np.argsort(combined_fitness[:self.pop_size])[-1:]
        self.best_individual = [self.population[best_index],combined_fitness[best_index]]

    
    #def diverse_evolve(self, parent_selection, crossover, mutation, survival_selection, n_parents=10, mutation_rate=0.2, track_diversity=True):
    def diverse_evolve(self, n_parents=10, mutation_rate=0.2, track_diversity=True):
    
        """Performs evolutionary algorithm with optional diversity tracking."""
        if track_diversity:
            self.track_diversity(self.population, generation=0)
        
        for generation in range(1, self.max_gens + 1):
            # Step 1: Evaluate fitness of the current population
            fitness_values = self.simulate()  # Method to evaluate population (assumed in Evolution)

            # Step 2: Select parents using the specified selection method
            parents = self.fitness_sharing(self.population, fitness_values, n_parents=n_parents)

            # Step 2: Generate Offspring via Crossover
            offspring = self.diverse_crossover(parents, n_offspring=self.pop_size)

            # Step 3: Mutate the Offspring
            mutated_offspring = []
            for child in offspring:
                mutated_child = self.diverse_mutation(child, mutation_rate=mutation_rate)  # Apply mutation
                mutated_offspring.append(mutated_child)

            mutated_offspring = np.array(mutated_offspring)

            # Step 4: Evaluate fitness of offspring
            offspring_fitness = self.simulate(mutated_offspring)

            # Step 5: Combine current population and offspring for survival selection
            combined_population = np.vstack((self.population, mutated_offspring))
            combined_fitness = np.concatenate((fitness_values, offspring_fitness))

            # Step 6: Survivor Selection (use the specified method for replacement)
            self.population = self.roulette_wheel_selection(combined_population, combined_fitness, n_parents=self.pop_size)

            # Step 6: Track fitness metrics
            self.track_fitness(combined_fitness[:self.pop_size])

            # Optional: Track genetic diversity metrics
            if track_diversity:
                self.track_diversity(self.population, generation)

        best_index = np.argsort(combined_fitness[:self.pop_size])[-1:]
        self.best_individual = [self.population[best_index],combined_fitness[best_index]]



    def calculate_population_diversity(self, population):
        """Calculates the genetic diversity of the population using mean pairwise Euclidean distance and variance."""
        num_individuals = len(population)
        pairwise_distances = []

        # Calculate mean pairwise distance
        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                distance = np.linalg.norm(population[i] - population[j])
                pairwise_distances.append(distance)

        mean_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0
        variance_diversity = np.var(population, axis=0).mean()

        return mean_pairwise_distance, variance_diversity

    def track_diversity(self, population, generation):
        """Tracks genetic diversity metrics for plotting later."""
        mean_pairwise_distance, variance_diversity = self.calculate_population_diversity(population)
        self.diversity_over_time.append({
            'generation': generation,
            'mean_pairwise_distance': mean_pairwise_distance,
            'variance_diversity': variance_diversity
        })

    def track_fitness(self, fitness_values):
        """Tracks the best, mean, and standard deviation of fitness for each generation."""
        best_fitness = np.max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        # Store these values to plot later
        self.best_fitness_over_time.append(best_fitness)
        self.mean_fitness_over_time.append(mean_fitness)
        self.std_fitness_over_time.append(std_fitness)



