import numpy as np
import tensorflow as tf
import random

def get_weights_vector(model):
    """Flatten all weights of the TF model into a 1D vector."""
    weights = model.get_weights()
    flat = np.concatenate([w.flatten() for w in weights])
    return flat


def set_weights_vector(model, flat_vector):
    """Load flattened weights back into the TF model."""
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0

    for shape in shapes:
        size = np.prod(shape)
        new_w = flat_vector[idx:idx+size].reshape(shape)
        new_weights.append(new_w)
        idx += size

    model.set_weights(new_weights)


# NOVELTY SEARCH
def euclidean(a, b):
    return np.linalg.norm(a - b)


def novelty_score(behaviour, behaviour_set, archive, k=10):
    """Compute novelty relative to all behaviours + archive."""
    all_behaviours = behaviour_set + archive
    distances = [euclidean(behaviour, b) for b in all_behaviours]
    distances.sort()
    return np.mean(distances[:k])


# GENETIC OPERATORS
def mutate(weights, mutation_rate=0.05, mutation_strength=0.4):
    """Gaussian mutation."""
    new = weights.copy()
    mask = np.random.rand(len(new)) < mutation_rate
    noise = np.random.normal(0, mutation_strength, size=len(new))
    new[mask] += noise[mask]
    return new


def crossover(parent1, parent2):
    """One-point crossover."""
    point = random.randint(0, len(parent1)-1)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child


def select_parents(population, novelty_scores, num_parents):
    """Tournament selection based on novelty."""
    parents = []
    for _ in range(num_parents):
        i, j = np.random.randint(0, len(population), 2)
        winner = i if novelty_scores[i] > novelty_scores[j] else j
        parents.append(population[winner])
    return parents



class GeneticNoveltyTrainer:
    def __init__(self, model_builder, pop_size=50, archive_prob=0.1):
        self.model_builder = model_builder
        self.pop_size = pop_size
        self.archive = []
        self.archive_prob = archive_prob

        # criar população inicial
        self.population = []
        temp_model = self.model_builder()
        base_dim = len(get_weights_vector(temp_model))

        for _ in range(pop_size):
            # vetor de pesos inicial aleatório
            w = np.random.uniform(-1, 1, size=base_dim)
            self.population.append(w)

    def evaluate_population(self, env_runner):
        """
        env_runner(model) -> returns behaviour characterization (np.array)
        """
        behaviours = []
        for ind in self.population:
            model = self.model_builder()
            set_weights_vector(model, ind)
            bc = env_runner(model)
            behaviours.append(bc)
        return behaviours

    def evolve(self, behaviours):
        novelty_scores = [
            novelty_score(b, behaviours, self.archive)
            for b in behaviours
        ]

        # parents
        parents = select_parents(self.population, novelty_scores, num_parents=self.pop_size)

        # reproduction
        new_population = []
        for _ in range(self.pop_size):
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        # occasional archive update
        for b in behaviours:
            if random.random() < self.archive_prob:
                self.archive.append(b)

        self.population = new_population

        return np.mean(novelty_scores), np.max(novelty_scores)

