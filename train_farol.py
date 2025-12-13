import json
import numpy as np
import matplotlib.pyplot as plt
from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
import os
from environments.environment_farol import FarolEnv
import random

# ============================================================
# 1. DEFINIÇÃO DOS MAPAS DA CURRICULUM
# ============================================================

def make_env(seed, dificuldade):
    """Cria um ambiente Farol com tamanho fixo mas layout dependente da seed."""
    return lambda: FarolEnv(
        tamanho=(50, 50),
        dificuldade=dificuldade,
        max_steps=300,
        seed=seed
    )

# ---------- FASE 1: mapas simples fixos ----------
fixed_maps = [make_env(s, dificuldade=1) for s in [101, 202, 303, 404, 505]]

# ---------- FASE 2: 20 mapas semi-aleatórios ----------
semi_random_maps = [make_env(s, dificuldade=2) for s in range(1000, 1020)]

# ---------- FASE 3: mapas totalmente aleatórios ----------
def fully_random_env():
    seed = random.randint(10000, 99999)
    dificuldade = random.choice([2, 3])
    return FarolEnv(
        tamanho=(50, 50),
        dificuldade=dificuldade,
        max_steps=300,
        seed=seed
    )


# ============================================================
# 2. FUNÇÃO DE CURRICULUM LEARNING
# ============================================================

def curriculum_env_factory(generation):
    """
    Retorna um ambiente dependendo da fase do treino.
    """
    if generation <= 20:
        # Fase 1 — só mapas fixos → rápido e estável
        return random.choice(fixed_maps)()

    elif generation <= 60:
        # Fase 2 — mistura mapas fixos e semi-aleatórios
        pool = fixed_maps + semi_random_maps
        return random.choice(pool)()

    else:
        # Fase 3 — mapas totalmente aleatórios
        return fully_random_env()


# Wrapper preciso para o EvolutionTrainer
def wrapper_env_factory():
    return curriculum_env_factory(wrapper_env_factory.generation)

wrapper_env_factory.generation = 1


# ============================================================
# 3. CRIAR TRAINER EVOLUTIVO
# ============================================================

trainer = EvolutionTrainer(
    model_builder=lambda: create_mlp(input_dim=8),  # 8 radares + 4 features
    pop_size=200,
    archive_prob=0.15,
    elite_fraction=0.05
)


# ============================================================
# 4. CICLO DE TREINO
# ============================================================

GENERATIONS = 200
MAX_STEPS = 350
history = []

for gen in range(1, GENERATIONS + 1):
    wrapper_env_factory.generation = gen
    print(f"\n=== Geração {gen} ===")

    h = trainer.train(
        env_factories=wrapper_env_factory,
        max_steps=MAX_STEPS,
        generations=1,
        episodes_per_individual=1,   # como pediste
        alpha=0.6,
        verbose=True,
        external_generation_offset=gen - 1
    )

    history.extend(h)


# ============================================================
# 5. GUARDAR HISTÓRICO
# ============================================================

os.makedirs("results/farol", exist_ok=True)

with open("results/farol/history_farol.json", "w") as f:
    json.dump(history, f, indent=4)

print("Histórico guardado em results/farol/history_farol.json")


# ============================================================
# 6. PLOTS
# ============================================================

gens     = [h["generation"] for h in history]
mean_fit = [h["mean_fitness"] for h in history]
max_fit  = [h["max_fitness"] for h in history]

plt.figure(figsize=(10,5))
plt.plot(gens, mean_fit, label="Mean Fitness", linewidth=2)
plt.plot(gens, max_fit, label="Max Fitness", linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Learning Curve — Fitness")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/farol/plot_fitness.png", dpi=300)
plt.close()


# ============================================================
# 7. GUARDAR CAMPEÃO
# ============================================================

ok, score = trainer.save_champion(
    "model/best_agent_farol",
    env_factory=lambda: curriculum_env_factory(9999),
    max_steps=MAX_STEPS,
    n_eval=20,
    threshold=-0.5
)

print("Champion saved:", ok, "| score:", score)