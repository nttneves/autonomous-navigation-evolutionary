# train_farol.py
import json
import numpy as np
import matplotlib.pyplot as plt
from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
from environments.environment_farol import FarolEnv
import random

# =====================================================
# 1. Curriculum Learning: lista de fábricas de ambientes
# =====================================================

def make_env(dificuldade, seed):
    """Devolve ambiente Farol com seed fixa."""
    return lambda: FarolEnv(
        tamanho=(50, 50),
        dificuldade=dificuldade,
        max_steps=300,
        seed=seed
    )

# ---------- curriculum ----------
curriculum_env_factories = []

# fase 1 – dificuldade 1
for s in range(100, 110):
    curriculum_env_factories.append(make_env(dificuldade=1, seed=s))

# fase 2 – dificuldade 2
for s in range(200, 210):
    curriculum_env_factories.append(make_env(dificuldade=2, seed=s))

# fase 3 – dificuldade 3
for s in range(300, 310):
    curriculum_env_factories.append(make_env(dificuldade=3, seed=s))

# shuffle opcional
random.shuffle(curriculum_env_factories)


# =====================================================
# 2. Função para escolher ambiente consoante a geração
# =====================================================

def curriculum_env_factory(generation):
    """
    Seleciona um ambiente baseado na fase do treino.
    Faz isto:
    - Geração 1–10   → dificuldade 1
    - Geração 11–20  → dificuldade 2
    - Geração 21–30  → dificuldade 3
    Depois mistura tudo livremente.
    """
    if generation <= 10:
        pool = curriculum_env_factories[0:10]
    elif generation <= 20:
        pool = curriculum_env_factories[10:20]
    elif generation <= 30:
        pool = curriculum_env_factories[20:30]
    else:
        pool = curriculum_env_factories  # mistura geral

    return random.choice(pool)()


# precisamos disto para passar ao trainer
def wrapper_env_factory():
    """Wrapper neutro: trainer só chama esta função."""
    return curriculum_env_factory(wrapper_env_factory.generation)

wrapper_env_factory.generation = 1  # atributo dinâmico


# =====================================================
# 3. Criar trainer
# =====================================================

trainer = EvolutionTrainer(
    model_builder=lambda: create_mlp(input_dim=10),
    pop_size=120,            # maior para generalização
    archive_prob=0.15,
    elite_fraction=0.05
)

# =====================================================
# 4. Treino com curriculum
# =====================================================

history = []

GENERATIONS = 35
MAX_STEPS = 250

for gen in range(1, GENERATIONS + 1):
    wrapper_env_factory.generation = gen
    print(f"\n=== CURRICULUM – Geração {gen} ===")

    h = trainer.train(
        env_factories=wrapper_env_factory,
        max_steps=MAX_STEPS,
        generations=1,
        episodes_per_individual=3,
        alpha=0.9,
        verbose=True,
        external_generation_offset=gen-1    # <-- AQUI
    )

    history.extend(h)


# =====================================================
# 5. Guardar histórico
# =====================================================

with open("results/farol/history_farol.json", "w") as f:
    json.dump(history, f, indent=4)

print("Histórico guardado em results/farol/history_farol.json")

# =====================================================
# 6. Curva de aprendizagem
# =====================================================

gens       = [h["generation"] for h in history]
mean_fit   = [h["mean_fitness"] for h in history]
max_fit    = [h["max_fitness"] for h in history]
mean_nov   = [h["mean_novelty"] for h in history]
max_nov    = [h["max_novelty"] for h in history]

plt.figure(figsize=(10,5))
plt.plot(gens, mean_fit, label="Mean Fitness", linewidth=2)
plt.plot(gens, max_fit, label="Max Fitness", linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Learning Curve – Fitness")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/farol/plot_fitness.png", dpi=300)
plt.close()

print("Curva guardada em results/farol/plot_fitness.png")

plt.figure(figsize=(10,5))
plt.plot(gens, mean_nov, label="Mean Novelty", linewidth=2)
plt.plot(gens, max_nov, label="Max Novelty", linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Novelty")
plt.title("Learning Curve – Novelty")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/farol/plot_novelty.png", dpi=300)
plt.close()

print("Gráfico Novelty guardado em results/farol/plot_novelty.png")

# =====================================================
# 7. Guardar campeão
# =====================================================

ok, score = trainer.save_champion(
    "model/best_agent_farol_curriculum.keras",
    env_factory=lambda: curriculum_env_factory(99999),
    max_steps=MAX_STEPS,
    n_eval=20,
    threshold=-0.5
)

print("Champion saved:", ok, "| score:", score)