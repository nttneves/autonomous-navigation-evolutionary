# algorithms/genetic.py 

import numpy as np
import random

# ===============================================================
# PESOS
# ===============================================================

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
        block = flat[idx:idx + size].reshape(shp)
        new_weights.append(block)
        idx += size

    model.set_weights(new_weights)


# ===============================================================
# NOVELTY SEARCH
# ===============================================================

def euclidean(a, b):
    return np.linalg.norm(a - b)


def novelty_score(bc, pop_bcs, archive, k=10):
    if bc is None:
        return 0.0

    all_bcs = [x for x in pop_bcs if x is not None] + \
              [x for x in archive if x is not None]

    if len(all_bcs) == 0:
        return 0.0

    dists = [euclidean(bc, x) for x in all_bcs]
    dists.sort()
    k = min(k, len(dists))

    return float(np.mean(dists[:k]))


# ===============================================================
# GENETIC OPERATORS
# ===============================================================

def mutate_gaussian_adaptive(w, base_rate=0.06, sigma=0.25, scale=1.0):
    """
    Mutação adaptativa: elites têm scale < 1, ruins têm scale > 1.
    """
    rate = base_rate * scale
    mask = np.random.rand(len(w)) < rate
    noise = np.random.normal(0, sigma * scale, size=len(w))

    new = w.copy()
    new[mask] += noise[mask]
    return new


def crossover_uniform(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)


def crossover_one_point(p1, p2):
    point = np.random.randint(1, len(p1)-1)
    return np.concatenate([p1[:point], p2[point:]])


def crossover_blend_safe(p1, p2, alpha=0.4):
    """
    Versão segura: evita explosão de pesos.
    """
    low = np.minimum(p1, p2)
    high = np.maximum(p1, p2)
    diff = high - low

    min_range = low - alpha * diff
    max_range = high + alpha * diff

    child = np.random.uniform(min_range, max_range)

    # clip seguro para evitar saturação do tanh
    return np.clip(child, -3.0, 3.0)


def pick_crossover():
    ops = [
        (crossover_uniform,      0.45),
        (crossover_one_point,    0.20),
        (crossover_blend_safe,   0.35)
    ]
    r = random.random()
    acc = 0.0
    for op, p in ops:
        acc += p
        if r <= acc:
            return op
    return crossover_uniform


# ===============================================================
# SELEÇÃO
# ===============================================================

def select_parents(pop, scores, n_parents):
    """
    Torneio NEAT-style: robusto e estável.
    """
    parents = []
    N = len(pop)

    for _ in range(n_parents):
        i, j = np.random.randint(0, N, 2)
        if scores[i] > scores[j]:
            parents.append(pop[i])
        else:
            parents.append(pop[j])

    return parents


# ===============================================================
# TRAINER
# ===============================================================

class GeneticNoveltyTrainer:
    def __init__(self, model_builder, pop_size=50,
                 archive_prob=0.1, elite_fraction=0.05, seed=42):

        random.seed(seed)

        self.model_builder = model_builder
        self.pop_size = pop_size
        self.archive_prob = archive_prob
        self.elite_n = max(1, int(pop_size * elite_fraction))
        self.archive = []

        # modelo temporário só para conhecer o tamanho do genoma
        temp = self.model_builder()
        dim = len(get_weights_vector(temp))

        # população inicial
        self.population = [
            np.random.normal(0, 1, size=dim) for _ in range(pop_size)
        ]

    def evolve(self, behaviours, fitnesses=None, alpha=0.7):
        """
        behaviours: lista de BCs (ou None)
        fitnesses: lista de fitness (pode ser None)
        alpha: peso da novidade no score final
        """

        # novelty
        novelty_arr = np.array([
            novelty_score(b, behaviours, self.archive, k=10)
            for b in behaviours
        ])

        # fitness normalizado
        if fitnesses is not None:
            F = np.array(fitnesses, dtype=np.float32)
            mn, mx = float(F.min()), float(F.max())
            if mx - mn > 1e-8:
                fit_norm = (F - mn) / (mx - mn)
            else:
                fit_norm = np.zeros_like(F)
        else:
            fit_norm = np.zeros(len(novelty_arr))

        # score final
        combined = alpha * novelty_arr + (1 - alpha) * fit_norm

        # -----------------------------------------------------------
        # Elitismo — agora baseado no *score final*, não só novelty
        # -----------------------------------------------------------
        elite_idx = np.argsort(combined)[::-1][:self.elite_n]
        elites = [self.population[i].copy() for i in elite_idx]

        # Seleção de pais por torneio
        parents = select_parents(self.population, combined, self.pop_size)

        # -----------------------------------------------------------
        # Nova geração
        # -----------------------------------------------------------
        new_pop = elites.copy()

        while len(new_pop) < self.pop_size:
            p1, p2 = random.sample(parents, 2)

            op = pick_crossover()
            child = op(p1, p2)

            # mutação adaptativa:
            # elites → scale = 0.3 (mais leve)
            # baixo score → scale = 1.5
            child_score = float(np.mean([combined[np.argmax(combined)]]))

            if child_score > np.percentile(combined, 75):
                scale = 0.3
            elif child_score < np.percentile(combined, 25):
                scale = 1.5
            else:
                scale = 1.0

            child = mutate_gaussian_adaptive(child, scale=scale)
            new_pop.append(child)

        # update archive
        for bc in behaviours:
            if bc is not None and random.random() < self.archive_prob:
                self.archive.append(bc.copy())

        self.population = new_pop

        return float(novelty_arr.mean()), float(novelty_arr.max())