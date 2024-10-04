import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from evoman.environment import Environment
from demo_controller import player_controller
from matplotlib import rcParams
import time


class Evolution:
    def __init__(self, enemy=[1], n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        self.enemy = enemy
        self.n_neurons = n_neurons
        self.low_weight = low_weight
        self.upp_weight = upp_weight
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.env = Environment(enemies=enemy, logs="off", savelogs="no", multiplemode="yes", player_controller=player_controller(n_neurons))

        # Initialize population
        self.population = self.create_population()
        self.best_individual = None

        # Initialize tracking variables for plotting
        self.best_fitness_over_time = []
        self.mean_fitness_over_time = []
        self.std_fitness_over_time = []
        self.diversity_over_time = []
        self.individual_fitness_over_time = []  # Track individual fitnesses for each generation

        print(f"Initialized Evolution with enemies {self.enemy}, population size {self.pop_size}, and max generations {self.max_gens}")

    def create_population(self):
        '''Creates a population of individuals with random weights within the range [low_weight, upp_weight]'''
        n_inputs = self.env.get_num_sensors()  # Number of input neurons
        n_outputs = 5  # Number of output neurons (actions)

        # Calculate number of weights
        input_to_hidden_weights = (n_inputs + 1) * self.n_neurons  # +1 for bias
        hidden_to_output_weights = (self.n_neurons + 1) * n_outputs  # +1 for bias
        self.n_vars = input_to_hidden_weights + hidden_to_output_weights

        print(f"Number of variables (weights) per individual: {self.n_vars}")
        return np.random.uniform(self.low_weight, self.upp_weight, (self.pop_size, self.n_vars))

    def simulate_individual(self, individual):
        '''Runs the simulation against each enemy separately and returns a list of fitness values'''
        fitness_values = []
        for enemy in self.enemy:
            # Create a new environment for each enemy to ensure correct settings
            env = Environment(enemies=[enemy], logs="off", savelogs="no", multiplemode="no", player_controller=player_controller(self.n_neurons))
            fitness, _, _, _ = env.play(pcont=individual)
            fitness_values.append(fitness)

        print(f"Simulated individual with fitness values: {fitness_values}")
        return fitness_values

    def simulate(self, population=None):
        '''Runs simulations for the given population and returns fitness values for each individual.'''
        if population is None:
            population = self.population

        all_fitness_values = []  # To store individual fitness values for all enemies
        geometric_fitness = []  # To store geometric mean fitness for each individual

        print("Simulating population...")

        # Step 1: Find the minimum fitness value across the population
        min_fitness_value = float('inf')
        for individual in population:
            fitness_values = self.simulate_individual(individual)
            min_fitness_value = min(min(fitness_values), min_fitness_value)  # Update the minimum fitness value

        # Step 2: Determine the shift value to ensure all fitness values are positive
        if min_fitness_value <= 0:
            shift_value = abs(min_fitness_value) + 1  # Shift by absolute value of minimum + 1 to ensure positivity
        else:
            shift_value = 0  # No shift needed if all values are positive

        print(f"Minimum fitness value found: {min_fitness_value}. Applying shift value: {shift_value}.")

        # Step 3: Calculate geometric mean for each individual using shifted fitness values
        for individual in population:
            fitness_values = self.simulate_individual(individual)  # Get fitness values for each enemy

            # Apply shift to fitness values
            shifted_fitness_values = [f + shift_value for f in fitness_values]

            # Calculate geometric mean on shifted values, handle NaNs gracefully
            try:
                geometric_fitness_value = gmean(shifted_fitness_values)
            except ValueError:
                geometric_fitness_value = float('nan')  # Assign NaN if gmean fails

            # Append both original and geometric fitness values
            all_fitness_values.append(fitness_values)
            geometric_fitness.append(geometric_fitness_value)

        print(f"Completed simulation for population. Geometric fitness values: {geometric_fitness}")
        return np.array(geometric_fitness), np.array(all_fitness_values)

    def evolve(self, parent_selection, crossover, mutation, survival_selection, n_parents=10, mutation_rate=0.2, track_diversity=True):
        """Performs evolutionary algorithm with optional diversity tracking."""
        if track_diversity:
            self.track_diversity(self.population, generation=0)

        for generation in range(1, self.max_gens + 1):
            print(f"Starting generation {generation}...")

            # Step 1: Evaluate fitness of the current population using geometric mean
            geometric_fitness, individual_fitness = self.simulate()  # Returns both geometric mean and individual fitness

            # Step 2: Select parents using the specified selection method
            parents = parent_selection(self.population, geometric_fitness, n_parents=n_parents)
            print(f"Selected parents for generation {generation}")

            # Step 3: Generate Offspring via Crossover
            offspring = crossover(parents, n_offspring=self.pop_size)
            print(f"Generated {len(offspring)} offspring")

            # Step 4: Mutate the Offspring
            mutated_offspring = np.array([mutation(child, mutation_rate) for child in offspring])
            print(f"Applied mutation to offspring")

            # Step 5: Evaluate fitness of offspring
            offspring_geometric_fitness, _ = self.simulate(mutated_offspring)
            print(f"Evaluated fitness of offspring")

            # Step 6: Combine current population and offspring for survival selection
            combined_population = np.vstack((self.population, mutated_offspring))
            combined_fitness = np.concatenate((geometric_fitness, offspring_geometric_fitness))
            print(f"Combined current population and offspring")

            # Step 7: Survivor Selection (use the specified method for replacement)
            self.population = survival_selection(combined_population, combined_fitness, n_parents=self.pop_size)
            print(f"Survivor selection complete for generation {generation}")

            # Step 8: Track fitness metrics
            self.track_fitness(combined_fitness[:self.pop_size], individual_fitness)

            # Optional: Track genetic diversity metrics
            if track_diversity:
                self.track_diversity(self.population, generation)

        best_index = np.argsort(combined_fitness[:self.pop_size])[-1]
        self.best_individual = [self.population[best_index], combined_fitness[best_index]]
        print(f"Best individual selected with fitness: {self.best_individual[1]}")

    def track_fitness(self, fitness_values, individual_fitness_values):
        """Tracks the best, mean, standard deviation of fitness, and stores individual fitness for each generation."""
        best_fitness = np.max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        # Store these values to plot later
        self.best_fitness_over_time.append(best_fitness)
        self.mean_fitness_over_time.append(mean_fitness)
        self.std_fitness_over_time.append(std_fitness)

        # Store individual fitness for each generation
        self.individual_fitness_over_time.append(individual_fitness_values)

        print(f"Tracked fitness: Best={best_fitness}, Mean={mean_fitness}, Std Dev={std_fitness}")

    def calculate_population_diversity(self, population):
        """Calculates the genetic diversity of the population using mean pairwise Euclidean distance and variance."""
        num_individuals = len(population)
        pairwise_distances = []

        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                distance = np.linalg.norm(population[i] - population[j])
                pairwise_distances.append(distance)

        mean_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0
        variance_diversity = np.var(population, axis=0).mean()

        print(f"Population diversity: Mean pairwise distance={mean_pairwise_distance}, Variance={variance_diversity}")
        return mean_pairwise_distance, variance_diversity

    def track_diversity(self, population, generation):
        """Tracks genetic diversity metrics for plotting later."""
        mean_pairwise_distance, variance_diversity = self.calculate_population_diversity(population)
        self.diversity_over_time.append({
            'generation': generation,
            'mean_pairwise_distance': mean_pairwise_distance,
            'variance_diversity': variance_diversity
        })

        print(f"Tracked diversity for generation {generation}: Mean distance={mean_pairwise_distance}, Variance={variance_diversity}")

    def plot_fitness(self):
        """Plots fitness metrics (best, mean, standard deviation, and individual fitness over time)."""
        generations = range(len(self.best_fitness_over_time))

        plt.figure(figsize=(12, 6))
        plt.plot(generations, self.best_fitness_over_time, label='Best Fitness')
        plt.plot(generations, self.mean_fitness_over_time, label='Mean Fitness')
        plt.fill_between(generations,
                         np.array(self.mean_fitness_over_time) - np.array(self.std_fitness_over_time),
                         np.array(self.mean_fitness_over_time) + np.array(self.std_fitness_over_time),
                         color='gray', alpha=0.3, label='Fitness Std Dev')

        # Plot individual fitness values for each generation
        for gen_idx, individual_fitness in enumerate(self.individual_fitness_over_time):
            plt.scatter([gen_idx] * len(individual_fitness), individual_fitness, color='red', s=10, alpha=0.5, label='Individual Fitness' if gen_idx == 0 else "")

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()




class Algorithm_Elitist(Evolution):
    def __init__(self, enemy=[1, 2, 3], n_neurons=10, low_weight=-1, upp_weight=1, pop_size=100, max_gens=30):
        super().__init__(enemy=enemy, n_neurons=n_neurons, low_weight=low_weight, upp_weight=upp_weight,
                         pop_size=pop_size, max_gens=max_gens)

    def elitist_selection(self, population, fitness, n_parents):
        sorted_indices = np.argsort(fitness)[-n_parents:]
        return population[sorted_indices]

    def uniform_mutation(self, individual, mutation_rate):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.uniform(self.low_weight, self.upp_weight)
        return individual

    def crossover(self, parents, n_offspring):
        num_parents = len(parents)
        genome_length = len(parents[0])
        num_cross_points = min(2 * num_parents - 1, genome_length // 2)
        offspring = np.zeros((n_offspring, genome_length))

        for i in range(n_offspring):
            crossover_points = np.sort(np.random.choice(range(1, genome_length), num_cross_points, replace=False))
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

Elitist = Algorithm_Elitist(
    enemy=[1, 2, 3],  # Multiple enemies
    n_neurons=10,
    low_weight=-1,
    upp_weight=1,
    pop_size=100,
    max_gens=30
)

# Running the evolution process
Elitist.evolve(
    parent_selection=Elitist.elitist_selection,
    crossover=Elitist.crossover,
    mutation=Elitist.uniform_mutation,
    survival_selection=Elitist.elitist_selection
)

# After evolution, plot the fitness over generations
Elitist.plot_fitness()