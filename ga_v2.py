import numpy as np

def selection(pop, fitness, num_parents):
    # We select the ones with the highest fitness
    sorted_indecies = np.argsort(fitness)[::-1] # we reverse array, because we want ascending
    top_indecies = sorted_indecies[:num_parents]
    parents = pop[top_indecies]
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty((offspring_size, parents.shape[1]))
    # The point at which crossover takes place between two parents
    crossover_part = int(parents.shape[1]/2)
    # Define all the offsprings
    for k in range(0, offspring_size, 2):
        # we assign random from the two parents
        random_choice = np.random.choice(parents.shape[1], crossover_part, replace=False)
        gene_selection = np.zeros(parents.shape[1])
        gene_selection[random_choice] = 1
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # Assign the correct parts of the parents to the offspring
        offspring[k, np.where(gene_selection == 0)[0]] = parents[parent1_idx, np.where(gene_selection == 0)[0]]
        offspring[k, np.where(gene_selection == 1)[0]] = parents[parent2_idx, np.where(gene_selection == 1)[0]]
        offspring[k+1, np.where(gene_selection == 1)[0]] = parents[parent1_idx, np.where(gene_selection == 1)[0]]
        offspring[k+1, np.where(gene_selection == 0)[0]] = parents[parent2_idx, np.where(gene_selection == 0)[0]]
    return offspring

def mutation(offspring_crossover, r_mut=0.2, v_mut=0.2):
    # Do if mutation stikes, must be changed if number of options exceed 2
    for idx in range(offspring_crossover.shape[0]):
        for gene_idx in range(offspring_crossover.shape[1]): 
            # Flip gene if mutation happens
            if (np.random.rand() < r_mut): 
                new_value = offspring_crossover[idx, gene_idx] + (np.random.rand()*2-1)*v_mut
                if (new_value < 0): 
                    new_value = 0
                elif (new_value >= 1): 
                    new_value = 0.99
                offspring_crossover[idx, gene_idx] = new_value 
                #offspring_crossover[idx, gene_idx] = 1-offspring_crossover[idx, gene_idx]# bit flip
    return offspring_crossover