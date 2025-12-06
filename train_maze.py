# train_maze.py
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
from environments.environment_maze import MazeEnv

# =====================================================
# 1. Curriculum Learning para o Maze (factories por dificuldade)
# =====================================================

def make_env(dificuldade):
    return lambda: MazeEnv(dificuldade=dificuldade, max_steps=200)

curriculum_env_factories = [
    make_env(0),
    make_env(1),
    make_env(2),
]

def curriculum_env_factory(curr_generation):
    if curr_generation <= 30:
        return curriculum_env_factories[0]()   # dif 0
    elif curr_generation <= 65:
        return curriculum_env_factories[1]()   # dif 1
    else:
        return curriculum_env_factories[2]()   # dif 2
    
def wrapper_env_factory():
    return curriculum_env_factory(wrapper_env_factory.generation)

wrapper_env_factory.generation = 1

# =====================================================
# 2. Trainer (parametros recomendados)
# =====================================================

trainer = EvolutionTrainer(
    model_builder=lambda: create_mlp(input_dim=12),
    pop_size=150,              # maior para maze
    archive_prob=0.15,
    elite_fraction=0.1
)

# =====================================================
# 3. Treino com curriculum (fazemos uma geração por loop para
#    poder controlar a fase do curriculum e guardar históricos corretos)
# =====================================================

GENERATIONS = 100
MAX_STEPS = 350
EPISODES_PER = 1

history = []

for gen in range(1, GENERATIONS + 1):
    wrapper_env_factory.generation = gen
    print(f"\n=== CURRICULUM MAZE – Geração {gen} ===")
    #if gen <= 30 or (gen > 75 and gen <= 115) or (gen > 150 and gen <= 210):
    #    alpha_value = 0.8# Foco na Novelty para encontrar novas soluções
    #else:
    #    alpha_value = 0.1 # Foco no Fitness para otimizar as soluções encontradas   
    if gen <= 15 or (gen > 30 and gen <= 45) or (gen >65 and gen <= 85):
        alpha_value = 0.8# Foco na Novelty para encontrar novas soluções
    else:
        alpha_value = 0.1 # Foco no Fitness para otimizar as soluções encontradas                

    h = trainer.train(
        env_factories=wrapper_env_factory,     # nota: nome env_factories no teu trainer
        max_steps=MAX_STEPS,
        generations=1,
        episodes_per_individual=EPISODES_PER,
        alpha=alpha_value,
        verbose=True,
        external_generation_offset=gen-1
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

ok, score = trainer.save_champion(
    "model/best_agent_maze",
    env_factory=lambda: curriculum_env_factory(0),
    max_steps=MAX_STEPS,
    n_eval=20,
    threshold=-10000.0
)

print("Champion saved:", ok, "| score:", score)