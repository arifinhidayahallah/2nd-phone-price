import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import  (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

phones_df = pd.read_csv('datasets/transformed_data/phones.csv')

X_data = phones_df.drop(columns=['price'])
y_data = phones_df['price']

X, X_test, y, y_test = train_test_split(
    X_data,
    y_data,
    test_size=.1
)

GEN_SIZE = 100
POP_SIZE = 40
N_FEATURES = len(X_data.columns)

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
            np.random.randint(50, 501),
            np.random.randint(3, 31),
            np.random.randint(2, 16),
            np.random.randint(1, 11),
            np.random.randint(0, 3)
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
        0:"sqrt",
        1:"log2",
        2: None
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
    n_splits=10,
    shuffle=True,
    random_state=42
)

def fitness_function(chromosome, X, y):
    feature_mask, params = decode(chromosome)
    
    if(feature_mask.sum()) == 0:
        return
    
    X_selected = X.loc[:, feature_mask]
    
    model = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(
        model,
        X_selected,
        y,
        cv=kf,
        scoring="r2"
    )
    
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
    
    
def boltzmann_selection(population, fitness, temperature):
    
    scaled = np.exp(
        (fitness-np.max(fitness)) / temperature
    )
    
    probability = scaled / scaled.sum()
    
    selected_index = np.random.choice(
        len(population),
        size=len(population),
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
    
    for i in range(N_FEATURES):
        if np.random.rand() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
            
    for i in range(N_FEATURES, len(parent1)):
        if np.random.rand() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
            
    return child1, child2


MUTATION_PROB = 0.2
SIGMA = 0.1

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
            chromosome[N_FEATURES ], 50, 500
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 1] = gaussian_mutation(
            chromosome[N_FEATURES + 1], 2, 50
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 2] = gaussian_mutation(
            chromosome[N_FEATURES + 2], 2, 20
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 3] = gaussian_mutation(
            chromosome[N_FEATURES + 3], 1, 20
        )
    if np.random.rand() < MUTATION_PROB:
        chromosome[N_FEATURES + 4] = np.random.randint(3)
        
    return chromosome



CROSSOVER_PROB = 0.8


for generation in range(GEN_SIZE):
    
    fitness = np.array([
        fitness_function(chromosome, X, y )
        for chromosome in population
    ])
    best_history.append(fitness.max())
    mean_history.append(fitness.mean())

    best_idx = np.argmax(fitness)
    if fitness[best_idx] > global_best_score:
        global_best_score = fitness[best_idx]
        global_best = population[best_idx].copy()

    parents = tournament_selection(population, fitness, k=3)
    print(f"Generation {generation}: {fitness.max():.4f}")
    offspring = []
    for j in range(0, POP_SIZE, 2):

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