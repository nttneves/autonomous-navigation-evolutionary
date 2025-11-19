import numpy as np
import tensorflow as tf
import random


def get_weights_vector(model):
    """
    Transform all weights of the TF model into a single flat vector.
    """
    weights = model.get_weights()
    if len(weights) == 0:
        raise ValueError("Model has no weights — check model_builder()")
    flat = np.concatenate([w.flatten() for w in weights])
    return flat


def set_weights_vector(model, flat_vector):
    """
    Load flattened weights back into the TF model.
    """
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0

    for shape in shapes:
        size = np.prod(shape)
        new_w = flat_vector[idx:idx+size].reshape(shape)
        new_weights.append(new_w)
        idx += size

    if idx != len(flat_vector):
        raise ValueError("Flat vector does not match model weight shapes.")

    model.set_weights(new_weights)


# NOVELTY SEARCH
def euclidean(a, b):
    return np.linalg.norm(a - b)


def novelty_score(behaviour, behaviour_set, archive, k=10):
    """
    Compute novelty relative to behaviours + archive.

    behaviour: np.array (BC)
    behaviour_set: list of np.array
    archive: list of np.array
    """

    # Garantia de que BC é válido
    if behaviour is None or not isinstance(behaviour, np.ndarray):
        return 0.0

    all_behaviours = []

    # adicionar população
    for b in behaviour_set:
        if b is not None:
            all_behaviours.append(b)

    # adicionar archive
    for b in archive:
        if b is not None:
            all_behaviours.append(b)

    if len(all_behaviours) == 0:
        return 0.0

    # distâncias ao comportamento atual
    distances = [euclidean(behaviour, b) for b in all_behaviours]
    distances.sort()

    k = min(k, len(distances))
    return float(np.mean(distances[:k]))


# GENETIC OPERATORS
def mutate(weights, mutation_rate=0.05, mutation_strength=0.4):
    """
    Gaussian mutation: with probability mutation_rate, each gene receives noise.
    """
    new = weights.copy()
    mask = np.random.rand(len(new)) < mutation_rate
    noise = np.random.normal(0, mutation_strength, size=len(new))
    new[mask] += noise[mask]
    return new


def crossover(parent1, parent2):
    """
    Uniform crossover: gene-by-gene choice.
    (Melhor que one-point para redes neuronais.)
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have same length.")
    mask = np.random.rand(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child


def select_parents(population, novelty_scores, num_parents):
    """
    Tournament selection based on novelty.
    """
    parents = []
    pop_size = len(population)

    for _ in range(num_parents):
        i, j = np.random.randint(0, pop_size, 2)

        if novelty_scores[i] > novelty_scores[j]:
            parents.append(population[i])
        else:
            parents.append(population[j])

    return parents


class GeneticNoveltyTrainer:
    """
    The class managing population, novelty calculation and evolution.
    """

    def __init__(self, model_builder, pop_size=50, archive_prob=0.1):
        """
        model_builder: callable → returns a new untrained model
        """
        self.model_builder = model_builder
        self.pop_size = pop_size
        self.archive_prob = archive_prob

        # archive para novelty
        self.archive = []

        # Criar população inicial de vetores
        temp_model = self.model_builder()
        initial_dim = len(get_weights_vector(temp_model))

        self.population = []
        for _ in range(pop_size):
            # inicialização: normal é melhor que uniforme
            w = np.random.normal(0, 1, size=initial_dim)
            self.population.append(w)

    # -----------------------------------------------------

    def evaluate_population(self, env_runner):
        """
        Avalia cada indivíduo da população.
        env_runner(model) → devolve um BC (np.array)

        Devolve lista de BCs.
        """

        behaviours = []

        for ind in self.population:
            model = self.model_builder()
            set_weights_vector(model, ind)

            try:
                bc = env_runner(model)
            except Exception as e:
                print("Erro ao avaliar indivíduo:", e)
                bc = None

            behaviours.append(bc)

        return behaviours

    # -----------------------------------------------------

    def evolve(self, behaviours):
        """
        Realiza um passo evolutivo completo:
        1. calcula novelty
        2. seleciona pais
        3. faz reprodução e mutação
        4. atualiza archive
        5. substitui população
        """

        novelty_scores = [
            novelty_score(b, behaviours, self.archive)
            for b in behaviours
        ]

        # seleção
        parents = select_parents(
            self.population,
            novelty_scores,
            num_parents=self.pop_size
        )

        # reprodução
        new_population = []
        for _ in range(self.pop_size):
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        # update archive ocasionalmente
        for bc in behaviours:
            if bc is not None and random.random() < self.archive_prob:
                self.archive.append(bc)

        self.population = new_population

        # devolver valores úteis para logging
        return float(np.mean(novelty_scores)), float(np.max(novelty_scores))