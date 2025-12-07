# algorithms/genetic.py
import numpy as np
import random

def get_weights_vector(model):
    weights = model.get_weights()
    if len(weights) == 0:
        raise ValueError("Model has no weights.")
    return np.concatenate([w.flatten() for w in weights])

def set_weights_vector(model, flat):
    if flat is None:
        return
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0
    for shp in shapes:
        size = int(np.prod(shp))
        block = flat[idx:idx+size].reshape(shp)
        new_weights.append(block)
        idx += size
    model.set_weights(new_weights)

def euclidean(a, b):
    return np.linalg.norm(a - b)

def novelty_score(bc, population_bcs, archive, k=10):
    if bc is None:
        return 0.0
    all_bcs = [x for x in population_bcs if x is not None] + [x for x in archive if x is not None]
    if len(all_bcs) == 0:
        return 0.0
    dists = [euclidean(bc, x) for x in all_bcs]
    dists.sort()
    k = min(k, len(dists))
    return float(np.mean(dists[:k]))

def mutate_gaussian(weights, mutation_rate=0.05, sigma=0.3):
    mask = np.random.rand(len(weights)) < mutation_rate
    noise = np.random.normal(0, sigma, size=len(weights))
    w = weights.copy()
    w[mask] += noise[mask]
    return w

def crossover_uniform(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)

def crossover_one_point(p1, p2):
    point = np.random.randint(1, len(p1)-1)
    return np.concatenate([p1[:point], p2[point:]])

def crossover_blend(p1, p2, alpha=0.4):
    low = np.minimum(p1, p2)
    high = np.maximum(p1, p2)
    diff = high - low
    min_range = low - alpha * diff
    max_range = high + alpha * diff
    return np.random.uniform(min_range, max_range)

def pick_crossover():
    ops = [(crossover_uniform, 0.4), (crossover_one_point, 0.2), (crossover_blend, 0.4)]
    r = random.random()
    acc = 0.0
    for op, p in ops:
        acc += p
        if r <= acc:
            return op
    return crossover_uniform

def select_parents(population, scores, n_parents):
    parents = []
    size = len(population)
    for _ in range(n_parents):
        i, j = np.random.randint(0, size, 2)
        if scores[i] > scores[j]:
            parents.append(population[i])
        else:
            parents.append(population[j])
    return parents

class GeneticNoveltyTrainer:
    def __init__(self, model_builder, pop_size=50, archive_prob=0.1, elite_fraction=0.05, seed=42):
        random.seed(seed)
        self.model_builder = model_builder
        self.pop_size = pop_size
        self.archive_prob = archive_prob
        self.elite_n = max(1, int(pop_size * elite_fraction))
        self.archive = []
        temp_model = self.model_builder()
        dim = len(get_weights_vector(temp_model))
        self.population = [np.random.normal(0, 1, size=dim) for _ in range(pop_size)]

    def evolve(self, behaviours, fitnesses=None, alpha=0.7):
        """
        behaviours: list of BC arrays
        fitnesses: optional list of floats (not normalized) -> used to combine with novelty
        alpha: weight for novelty in combined score (0..1). combined = alpha*novelty + (1-alpha)*fitness_norm
        Returns: (mean_novelty, max_novelty)
        """
        novelty_scores = [novelty_score(b, behaviours, self.archive, k=10) for b in behaviours]

        # normalize fitnesses to [0,1] if provided
        if fitnesses is not None:
            arr = np.array(fitnesses, dtype=np.float32)
            # some fitnesses can be negative; shift
            mn, mx = float(arr.min()), float(arr.max())
            if mx - mn > 1e-8:
                fitness_norm = (arr - mn) / (mx - mn)
            else:
                fitness_norm = np.zeros_like(arr)
        else:
            fitness_norm = np.zeros(len(novelty_scores), dtype=np.float32)

        # combined scores for selection
        novelty_arr = np.array(novelty_scores, dtype=np.float32)
        combined = alpha * novelty_arr + (1.0 - alpha) * fitness_norm

        # elitism by novelty (keep most novel to preserve archive exploration)
        idx_sorted = np.argsort(novelty_arr)[::-1]
        elites = [self.population[i] for i in idx_sorted[:self.elite_n]]

        # parent selection uses combined score (tournament)
        parents = select_parents(self.population, combined, n_parents=self.pop_size)

        new_population = elites.copy()
        while len(new_population) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            op = pick_crossover()
            child = op(p1, p2)
            child = mutate_gaussian(child, mutation_rate=0.07, sigma=0.25)
            new_population.append(child)

        # update archive probabilistically
        for bc in behaviours:
            if bc is not None and random.random() < self.archive_prob:
                self.archive.append(bc.copy())

        self.population = new_population

        return float(novelty_arr.mean()), float(novelty_arr.max())