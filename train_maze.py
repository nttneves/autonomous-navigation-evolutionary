# train_maze.py
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
from environments.environment_maze import MazeEnv

# =====================================================
# 1. Curriculum Learning (mantemos dificuldades e seeds fixas)
# =====================================================

def make_env(dificuldade):
    """
    Mantém comportamento original:
    - dificuldade 0 → seed 42
    - dificuldade 1 → seed 150
    - dificuldade 2 → seed 456
    Cada factory devolve um MazeEnv sempre com o mesmo mapa.
    """
    return lambda: MazeEnv(dificuldade=dificuldade, max_steps=200)

curriculum_env_factories = [
    make_env(0),
    make_env(1),
    make_env(2),
]


# =====================================================
# 2. Trainer
# =====================================================

trainer = EvolutionTrainer(
    model_builder=lambda: create_mlp(input_dim=12),
    pop_size=300,
    archive_prob=0.15,
    elite_fraction=0.1
)


# =====================================================
# 3. Treino (USAMOS SEMPRE MULTI-ENV: os 3 mapas fixos)
#    Isto impede overfitting ao último mapa visto.
# =====================================================

GENERATIONS = 2000
MAX_STEPS = 450
EPISODES_PER = 2   # melhor do que 1 sem aumentar muito custo

history = []

for gen in range(1, GENERATIONS + 1):

    print(f"\n=== CURRICULUM MAZE – Geração {gen} ===")

    # curriculum afeta apenas alpha (exploração vs exploração)
    if gen <= 1000:
        alpha_value = 0.8   # exploração
    elif gen <= 1500:
        alpha_value = 0.5   # transição
    else:
        alpha_value = 0.2   # exploração mínima, foco total no fitness

    # Passamos SEMPRE os 3 mapas → trainer usa evaluate_population_multi
    h = trainer.train(
        env_factories=curriculum_env_factories, 
        max_steps=MAX_STEPS,
        generations=1,
        episodes_per_individual=EPISODES_PER,
        alpha=alpha_value,
        verbose=True,
        external_generation_offset=gen - 1
    )

    history.extend(h)


# =====================================================
# 4. Guardar histórico
# =====================================================

with open("results/maze/history_maze.json", "w") as f:
    json.dump(history, f, indent=4)

print("Histórico guardado em results/maze/history_maze.json")


# =====================================================
# 5. Curva de aprendizagem – FITNESS
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
plt.savefig("results/maze/plot_fitness.png", dpi=300)
plt.close()

print("Curva Fitness guardada em results/maze/plot_fitness.png")


# =====================================================
# 6. Curva de aprendizagem – NOVELTY
# =====================================================

plt.figure(figsize=(10,5))
plt.plot(gens, mean_nov, label="Mean Novelty", linewidth=2)
plt.plot(gens, max_nov,  label="Max Novelty",  linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Novelty")
plt.title("Learning Curve – Novelty")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/maze/plot_novelty.png", dpi=300)
plt.close()

print("Gráfico Novelty guardado em results/maze/plot_novelty.png")


# =====================================================
# 7. Guardar campeão
# =====================================================

# Avaliamos o campeão usando *uma factory* (podes escolher qualquer dificuldade).
ok, score = trainer.save_champion(
    "model/best_agent_maze",
    env_factory=lambda: MazeEnv(dificuldade=2, max_steps=MAX_STEPS),  # seed fixa como pediste
    max_steps=MAX_STEPS,
    n_eval=20,
    threshold=-10000.0
)

print("Champion saved:", ok, "| score:", score)