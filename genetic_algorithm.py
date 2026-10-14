import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics  import  (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#DATA SPLITTING
TEST_SIZE = .2

# RANDOM STATE
RANDOM_STATE = 42

#SELECTION METHOD
SELECTION_METHOD = 'roulette'

# RANDOM FOREST HYPERPARAMETER
N_ESTIMATORS = [100, 1000]
MAX_DEPTH = [5, 15]
MIN_SAMPLE_SPLIT = [5, 10]
MIN_SAMPLE_LEAF = [1, 10]
MAX_FEATURES = [0, 2]

# CROSS VALIDATION
CROSS_VALIDATION = 10

# GENETIC ALGORITHM HYPERPARAMETERS
GEN_SIZE = 50
POP_SIZE = 20
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.1
SIGMA = 0.1
ELITISM_COUNT = 2

phones_df = pd.read_csv('datasets/transformed_data/phones.csv')
X_data = phones_df.drop(columns=['price'])
y_data = phones_df['price']

N_FEATURES = len(X_data.columns)

X, X_test, y, y_test = train_test_split(
    X_data,
    y_data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

population = []

for _ in range(POP_SIZE):
    feature_mask = np.random.randint(
        0,
        2,
        N_FEATURES
    )

    if feature_mask.sum() == 0:
        feature_mask[np.random.randint(N_FEATURES)] = 1

    chromosome = np.concatenate([
        feature_mask,
        [
            np.random.randint(N_ESTIMATORS[0], N_ESTIMATORS[1]),
            np.random.randint(MAX_DEPTH[0], MAX_DEPTH[1]),
            np.random.randint(MIN_SAMPLE_SPLIT[0], MIN_SAMPLE_SPLIT[1]),
            np.random.randint(MIN_SAMPLE_LEAF[0], MIN_SAMPLE_LEAF[1]),
            np.random.randint(MAX_FEATURES[0], MAX_FEATURES[1])
        ]]
    )

    population.append(chromosome)

population = np.array(population,dtype=object)


def decode(chromosome):

    feature_mask = chromosome[:N_FEATURES].astype(bool)

    n_estimators = int(chromosome[N_FEATURES])
    max_depth = int(chromosome[N_FEATURES + 1])
    min_split = int(chromosome[N_FEATURES + 2])
    min_leaf = int(chromosome[N_FEATURES + 3])
    max_features_gene = int(chromosome[N_FEATURES + 4])

    max_features_map = {
        0: 1.0,
        1:"sqrt",
        2:"log2",
    }
    
    params = {
        "n_estimators" : n_estimators,
        "max_depth" : max_depth,
        "min_samples_split": min_split,
        "min_samples_leaf": min_leaf,
        "max_features": max_features_map[max_features_gene]
    }


    return feature_mask, params


mask, params = decode(population[0])

mask, params


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
global_best_score = 0
global_best_chromosome = []

kf = KFold(
    n_splits=CROSS_VALIDATION,
    shuffle=True,
    random_state=RANDOM_STATE
)


def fitness_function(chromosome, X, y):
    feature_mask, params = decode(chromosome)

    if(feature_mask.sum()) == 0:
        return

    X_selected = X.loc[:, feature_mask]
    model = RandomForestRegressor(
        **params,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_selected,
        y,
        cv=kf,
        scoring="r2"
    )
    # print(scores)
    return np.mean(scores)

best_history = []
mean_history = []


def tournament_selection(population, fitness, k=3):
    selected = []

    pop_size = len(population)

    for _ in range(pop_size):

        idx = np.random.choice(
            pop_size,
            size=k,
            replace=False
        )

        tournament_fitness = fitness[idx]

        winner = idx[np.argmax(tournament_fitness)]

        selected.append(population[winner].copy())

    return np.array(selected, dtype=object)

def roulette_selection(population, fitness):

    adjusted = fitness - fitness.min() + 1e-10

    probability = adjusted / adjusted.sum()

    selected_index = np.random.choice(
        len(population),
        size = len(population),
        replace=True,
        p=probability
    )

    return np.array(
        [population[i].copy() for i in selected_index],
        dtype=object
    )

def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()

    crossover_point = np.random.randint(1, len(child1))
    
    child1 = np.concatenate((child1[:crossover_point], child2[crossover_point:]))
    child2 = np.concatenate((child2[:crossover_point], child1[crossover_point:]))
    # for i in range(N_FEATURES):
    #     if np.random.rand() < 0.5:
    #         child1[i], child2[i] = child2[i], child1[i]

    # for i in range(N_FEATURES, len(parent1)):
    #     if np.random.rand() < 0.5:
    #         child1[i], child2[i] = child2[i], child1[i]

    return child1, child2

def gaussian_mutation(value,  lower, upper):
    std = SIGMA * (upper - lower)
    value += np.random.normal(0, std)
    value = np.clip(value, lower, upper)

    return int(round(value))

def mutate(chromosome):
    chromosome = chromosome.copy()

    for i in range(N_FEATURES):
        if np.random.rand() < MUTATION_PROB:
            chromosome[i] = 1 - chromosome[i]

    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES ] = gaussian_mutation(
            chromosome[N_FEATURES ], N_ESTIMATORS[0], N_ESTIMATORS[1]
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 1] = gaussian_mutation(
            chromosome[N_FEATURES + 1], MAX_DEPTH[0], MAX_DEPTH[1]
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 2] = gaussian_mutation(
            chromosome[N_FEATURES + 2], MIN_SAMPLE_SPLIT[0], MIN_SAMPLE_SPLIT[1]
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 3] = gaussian_mutation(
            chromosome[N_FEATURES + 3], MIN_SAMPLE_LEAF[0], MIN_SAMPLE_LEAF[1]
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 4] = np.random.randint(MAX_FEATURES[0], MAX_FEATURES[1])

    return chromosome

import os
# if TEST_SIZE == .1:
#     test_label = 10
# elif TEST_SIZE == .2:
#     test_label = 20
    
PATH_LOGS = f"logs/{SELECTION_METHOD}_{str(TEST_SIZE).replace('0.','')}0_percent_test/{SELECTION_METHOD}_selection-{str(CROSSOVER_PROB).replace(".", "_")}-logs.csv"

def save_logs(
    generation,
    # population,
    fitness,
    global_best,
    global_best_score,
    time_start,
    overall_start,
    execution_finish
):
    df_logs = pd.DataFrame([{
        "generation":generation,
        # "population":population,
        "fitness":fitness,
        "global_best":global_best,
        "global_best_score":global_best_score,
        "time_start":time_start,
        "overall_start":overall_start,
        "execution_finish":execution_finish,
    }])

    df_logs.to_csv(
        PATH_LOGS,
        mode="a",
        header=not os.path.exists(PATH_LOGS),
        index=False
    )


import time
import concurrent.futures
if __name__ == '__main__':
    print(f"{str(TEST_SIZE).replace('0.','')}0 Percent Test - {SELECTION_METHOD} Selection ({str(CROSSOVER_PROB)})")
    overall_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for generation in range(GEN_SIZE):
            generation_start = time.perf_counter()
            
            print(f"Generation {generation + 1}")
            
            print(f"Executing Fitness Function")
            
            futures = [
                executor.submit(fitness_function, chromosome, X, y)
                for chromosome in population
            ]
            
            fitness = np.array([f.result() for f in futures])
            # fitness = np.array([
            #     fitness_function(chromosome, X, y )
            #     for chromosome in population
            # ])
            best_history.append(fitness.max())
            mean_history.append(fitness.mean())

            best_idx = np.argmax(fitness)
            if fitness[best_idx] > global_best_score:
                global_best_score = fitness[best_idx]
                global_best = population[best_idx].copy()

            print(f"Executing Tournament Selection Function")
            if SELECTION_METHOD == 'tournament':
                parents = tournament_selection(population, fitness)
            elif SELECTION_METHOD == 'roulette':
                parents = roulette_selection(population, fitness)
                
            generation_time = time.perf_counter() - generation_start
            elapsed_time = time.perf_counter() - overall_start
            
            print(f"Executing Selection Function")
            save_logs(
                generation + 1,
                # population,
                fitness,
                global_best,
                global_best_score,
                generation_start,
                overall_start,
                time.perf_counter()
            )
            
            print(f"Executing Crossover Function")
            offspring = []
            sorted_indices = np.argsort(fitness)[::-1]
            for j in range(ELITISM_COUNT):
                best_index = sorted_indices[j]
                offspring.append(population[best_index])
                
            for j in range(0, POP_SIZE - (1 + ELITISM_COUNT), 2):

                parent1 = parents[j]
                parent2 = parents[j+1]

                if np.random.rand() < CROSSOVER_PROB:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()

                child1 = mutate(child1)
                child2 = mutate(child2)

                offspring.extend([child1, child2])
            # offspring.append([child1, child2])
            population = np.array(offspring, dtype=object)

        overall_end = time.perf_counter()

        print(f"Time Execution All Generation: {(overall_end - overall_start):.2f}")

        global_best

        global_best_score

        global_best_score

