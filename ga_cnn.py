import numpy as np
import json

from fashion_mnist_cnn import *
from ga import *


"""
The target is to maximize the accuracy of the cnn-mnist model:
    We start by having a few decisions related to the model. 
    // Model hyper-parameters
    x1 = 32-128 filters in each CNN layer
    x2 = 50-100 neurons in first dense layer
    x3 = 0-0.2 dropout 
    x4 = learning rate 0.005 or 0.05 
    x5 = kernel size cnn 2-4
    
    // Data augmentation hyper-parameters
    x6 = rotation_range: 0-30
    x7 = width_shift_range: 0.0-0.3,
    x8 = height_shift_range: 0.0-0.3
    x9 = horizontal_flip: 0 or 1
    
    We will use a genetic algorithm to find the best combinations of these hyperparameters. 
    The fitness-function is based on model accuracy on test-dataset. 
"""
num_weights = 9

# solutions per populations and number of mating parents
sol_per_pop = 8
num_parents = 4

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = np.random.uniform(low=0, high=1, size=pop_size)
print(new_population)


best_outputs = []
num_generations = 10
generation_results = []
checkpoint_file = "fit_pop.json"


for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = evaluate_model_ga(new_population)
    print("Fitness")
    print(fitness)
    
    # save the results into a list and do intemediate saving
    generation_results.append({"fitness": fitness, "population": [chrom.tolist() for chrom in new_population]})
    with open(checkpoint_file, "w") as f: 
        json.dump(generation_results, f)
    
    # Find best result from gen x
    generation_best = (np.max(fitness), new_population[np.where(fitness == np.max(fitness))[0]])
    best_outputs.append(generation_best)
    # The best result in the current iteration.
    print("Best result : ", generation_best)
    
    # Selecting the best parents in the population for mating.
    parents = selection(new_population, fitness, num_parents)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, sol_per_pop)
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, r_mut=0.2)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population = offspring_mutation