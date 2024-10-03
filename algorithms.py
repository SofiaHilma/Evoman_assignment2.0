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



class Evolution:
    def __init__(self, enemy=[1], n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        self.enemy = enemy
        self.n_neurons = n_neurons
        self.low_weight = low_weight
        self.upp_weight = upp_weight
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.env = Environment(enemies=enemy, logs="off", savelogs="no", player_controller=player_controller(n_neurons))
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
    
    def evolve(self, parent_selection, crossover, mutation, survival_selection, n_parents=10, mutation_rate=0.2,
               track_diversity=True):
        """Performs evolutionary algorithm with optional diversity tracking."""
        if track_diversity:
            self.track_diversity(self.population, generation=0)
        
        for generation in range(1, self.max_gens + 1):
            # Step 1: Evaluate fitness of the current population
            fitness_values = self.simulate()  # Method to evaluate population (assumed in Evolution)

            # Step 2: Select parents using the specified selection method
            parents = parent_selection(self.population, fitness_values, n_parents=n_parents)

            # Step 2: Generate Offspring via Crossover
            offspring = crossover(parents, n_offspring=self.pop_size)

            # Step 3: Mutate the Offspring
            mutated_offspring = []
            for child in offspring:
                mutated_child = mutation(child, mutation_rate=mutation_rate)  # Apply mutation
                mutated_offspring.append(mutated_child)

            mutated_offspring = np.array(mutated_offspring)

            # Step 4: Evaluate fitness of offspring
            offspring_fitness = self.simulate(mutated_offspring)

            # Step 5: Combine current population and offspring for survival selection
            combined_population = np.vstack((self.population, mutated_offspring))
            combined_fitness = np.concatenate((fitness_values, offspring_fitness))

            # Step 6: Survivor Selection (use the specified method for replacement)
            self.population = survival_selection(combined_population, combined_fitness, n_parents=self.pop_size)

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



class Algorithm_Elitist(Evolution):
    def __init__(self, enemy=[1], n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        super().__init__(enemy=enemy, n_neurons=n_neurons, low_weight=low_weight,
                         upp_weight=upp_weight, pop_size=pop_size, max_gens=max_gens)

    def elitist_selection(self, population, fitness, n_parents):
        """
        Selects the n_parents best individuals based on fitness.
        Can be used for both parent selection and survival selection.
        """
        sorted_indices = np.argsort(fitness)[-n_parents:]
        return population[sorted_indices]

    ### 2. Tournament Selection Method ###
    def tournament_selection(self, population, fitness, n_parents, tournament_size=3):
        selected = []

        for _ in range(n_parents):
            tournament_indices = np.random.randint(0, len(population), size=tournament_size)
            best_in_tournament = tournament_indices[np.argmax(fitness[tournament_indices])]
            selected.append(population[best_in_tournament])
        return np.array(selected)

    def uniform_mutation(self, individual, mutation_rate):

        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                # Replace the gene with a new random value within the valid range
                individual[i] = np.random.uniform(self.low_weight, self.upp_weight)
        return individual

    # Multi-parent-multi-point crossover
    def crossover(self, parents, n_offspring):
        """
        Performs a multi-parent-multi-point crossover.
        Returns the required number of children.
        """
        num_parents = len(parents)
        genome_length = len(parents[0])

        num_cross_points = min(2 * num_parents - 1, genome_length // 2)

        offspring = np.zeros((n_offspring, genome_length))

        for i in range(n_offspring):
            crossover_points = np.sort(np.random.choice(range(1, genome_length), num_cross_points, replace=False))
            # Ensure that crossover points are at least a few units apart (optional):
            crossover_points = np.unique(np.clip(crossover_points, 1, genome_length - 2))

            current_parent_index = np.random.randint(len(parents))
            current_parent = parents[current_parent_index]

            child = np.zeros(genome_length)
            prev_point = 0

            for point in crossover_points:
                child[prev_point:point] = current_parent[prev_point:point]
                prev_point = point
                current_parent_index = (current_parent_index + 1) % len(parents)
                current_parent = parents[current_parent_index]

            child[prev_point:] = current_parent[prev_point:]
            offspring[i] = child

        return offspring



class Algorithm_Diverse(Evolution):
    def __init__(self, enemy=[1], n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        # Call the superclass __init__ method to initialize population and other attributes
        super().__init__(enemy=enemy, n_neurons=n_neurons, low_weight=low_weight,
                         upp_weight=upp_weight, pop_size=pop_size, max_gens=max_gens)
        self.diversity_over_time = []

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

    def roulette_wheel_selection(self, population, fitness, n_parents):
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
        return population[selected_indices]


    def mutation(self, individual, mutation_rate, scale=0.4):
        # Implement mutation with a Cauchy distribution
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] += np.random.standard_cauchy() * scale
                individual[i] = np.clip(individual[i], self.low_weight, self.upp_weight)
        return individual

    def crossover(self, parents, n_offspring):
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

