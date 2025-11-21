#genetic.py
import numpy as np
import random
import tensorflow as tf


def get_weights_vector(model):
    """Flatten all TF model weights."""
    weights = model.get_weights()
    if len(weights) == 0:
        raise ValueError("Model has no weights.")
    return np.concatenate([w.flatten() for w in weights])


def set_weights_vector(model, flat):
    """Load flattened weights into model."""
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0

    for shp in shapes:
        size = np.prod(shp)
        block = flat[idx:idx+size].reshape(shp)
        new_weights.append(block)
        idx += size

    model.set_weights(new_weights)


def euclidean(a, b):
    return np.linalg.norm(a - b)


def novelty_score(bc, population_bcs, archive, k=10):
    if bc is None:
        return 0.0

    all_bcs = [x for x in population_bcs if x is not None] + \
              [x for x in archive if x is not None]

    if len(all_bcs) == 0:
        return 0.0

    dists = [euclidean(bc, x) for x in all_bcs]
    dists.sort()

    k = min(k, len(dists))
    return float(np.mean(dists[:k]))


def mutate_gaussian(weights, mutation_rate=0.05, sigma=0.3):
    """Gaussian noise applied gene-wise."""
    mask = np.random.rand(len(weights)) < mutation_rate
    noise = np.random.normal(0, sigma, size=len(weights))
    w = weights.copy()
    w[mask] += noise[mask]
    return w


def crossover_uniform(p1, p2):
    """Gene-by-gene uniform crossover."""
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)


def crossover_one_point(p1, p2):
    """Single-cut crossover."""
    point = np.random.randint(1, len(p1)-1)
    return np.concatenate([p1[:point], p2[point:]])


def crossover_blend(p1, p2, alpha=0.4):
    """BLX-α crossover – excelente para pesos contínuos."""
    low = np.minimum(p1, p2)
    high = np.maximum(p1, p2)
    diff = high - low

    min_range = low - alpha * diff
    max_range = high + alpha * diff

    return np.random.uniform(min_range, max_range)


def pick_crossover():
    """Escolhe um operador aleatório com probabilidades equilibradas."""
    ops = [
        (crossover_uniform, 0.4),
        (crossover_one_point, 0.2),
        (crossover_blend, 0.4),
    ]
    r = random.random()
    acc = 0.0
    for op, p in ops:
        acc += p
        if r <= acc:
            return op
    return crossover_uniform


def select_parents(population, novelty_scores, n_parents):
    """Tournament selection + bias para diversidade."""
    parents = []
    size = len(population)

    for _ in range(n_parents):
        i, j = np.random.randint(0, size, 2)

        # desempate = indivíduo com mais variância nos pesos (mais diverso)
        if novelty_scores[i] > novelty_scores[j]:
            parents.append(population[i])
        elif novelty_scores[j] > novelty_scores[i]:
            parents.append(population[j])
        else:
            parents.append(
                population[i] if np.std(population[i]) > np.std(population[j]) else population[j]
            )

    return parents


class GeneticNoveltyTrainer:
    """
    Genetic Algorithm com Novelty Search + múltiplos crossovers,
    elitismo e mutação gaussiana.
    """

    def __init__(self, model_builder, pop_size=50, archive_prob=0.1, elite_fraction=0.05):
        self.model_builder = model_builder
        self.pop_size = pop_size
        self.archive_prob = archive_prob

        # elitismo
        self.elite_n = max(1, int(pop_size * elite_fraction))

        self.archive = []

        # dimensão dos pesos
        temp_model = self.model_builder()
        dim = len(get_weights_vector(temp_model))

        # inicialização normal
        self.population = [
            np.random.normal(0, 1, size=dim) for _ in range(pop_size)
        ]

    # -----------------------------------------------------

    def evolve(self, behaviours):
        """Um passo evolutivo completo."""

        # novelty para cada indivíduo
        novelty_scores = [
            novelty_score(b, behaviours, self.archive, k=10)
            for b in behaviours
        ]

        # -----------------------------------------------------
        # 1. ELITISMO — mantém os melhores intactos
        # -----------------------------------------------------
        idx_sorted = np.argsort(novelty_scores)[::-1]
        elites = [self.population[i] for i in idx_sorted[:self.elite_n]]

        # -----------------------------------------------------
        # 2. Seleção de pais
        # -----------------------------------------------------
        parents = select_parents(
            self.population,
            novelty_scores,
            n_parents=self.pop_size
        )

        # -----------------------------------------------------
        # 3. Reprodução
        # -----------------------------------------------------
        new_population = elites.copy()

        while len(new_population) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            op = pick_crossover()
            child = op(p1, p2)
            child = mutate_gaussian(child)
            new_population.append(child)

        # -----------------------------------------------------
        # 4. Atualizar archive
        # -----------------------------------------------------
        for bc in behaviours:
            if bc is not None and random.random() < self.archive_prob:
                self.archive.append(bc)

        self.population = new_population

        # resultados de logging
        return float(np.mean(novelty_scores)), float(np.max(novelty_scores))