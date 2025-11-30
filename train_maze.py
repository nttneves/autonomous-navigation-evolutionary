# train_maze.py
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
from environments.environment_maze import MazeEnv


# =====================================================
# 1. Curriculum Learning para o Maze
# =====================================================

def make_env(dificuldade):
    """
    MazeEnv já define seeds fixas por dificuldade:
        diff 0 → seed 42
        diff 1 → seed 150
        diff 2 → seed 456
    """
    return lambda: MazeEnv(dificuldade=dificuldade, max_steps=600)


# lista de fábricas
curriculum_env_factories = [
    make_env(0),  # dificuldade 0  (seed=42)
    make_env(1),  # dificuldade 1  (seed=150)
    make_env(2),  # dificuldade 2  (seed=456)
]

# =====================================================
# 2. Curriculum Learning progressivo
# =====================================================

def curriculum_env_factory(curr_generation):
    """
    Define qual dificuldade usar em cada parte do treino:

      Geração 1–10  → dificuldade 0 (mais fácil)
      Geração 11–20 → dificuldade 1
      Geração 21–35 → dificuldade 2 (final)
      Depois disso → mistura tudo, para robustez
    """
    if curr_generation <= 10:
        return curriculum_env_factories[0]()   # dif 0
    elif curr_generation <= 20:
        return curriculum_env_factories[1]()   # dif 1
    elif curr_generation <= 35:
        return curriculum_env_factories[2]()   # dif 2
    else:
        return random.choice(curriculum_env_factories)()


# wrapper usado pelo trainer
def wrapper_env_factory():
    return curriculum_env_factory(wrapper_env_factory.generation)

wrapper_env_factory.generation = 1


# =====================================================
# 3. Criar trainer
# =====================================================

trainer = EvolutionTrainer(
    model_builder=lambda: create_mlp(input_dim=10),
    pop_size=200,              
    archive_prob=0.15,
    elite_fraction=0.05
)


# =====================================================
# 4. Treino com curriculum
# =====================================================

GENERATIONS = 50
MAX_STEPS = 800

history = []

for gen in range(1, GENERATIONS + 1):
    wrapper_env_factory.generation = gen
    print(f"\n=== CURRICULUM MAZE – Geração {gen} ===")

    h = trainer.train(
        env_factories=wrapper_env_factory,
        max_steps=MAX_STEPS,
        generations=1,              # fazemos uma geração de cada vez
        episodes_per_individual=3,  # mais episódios = mais robusto
        alpha=0.9,
        verbose=True,
        external_generation_offset= gen - 1
    )

    history.extend(h)


# =====================================================
# 5. Guardar histórico
# =====================================================

with open("results/history_maze.json", "w") as f:
    json.dump(history, f, indent=4)

print("Histórico guardado em results/history_maze.json")


# =====================================================
# 6. Curva de Aprendizagem
# =====================================================

gens = [h["generation"] for h in history]
mean_fit = [h["mean_fitness"] for h in history]
max_fit = [h["max_fitness"] for h in history]

plt.figure(figsize=(10,5))
plt.plot(gens, mean_fit, label="Mean Fitness", linewidth=2)
plt.plot(gens, max_fit, label="Max Fitness", linewidth=2)
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Curva de Aprendizagem – Maze (Curriculum Learning)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curve_maze.png", dpi=300)
plt.close()

print("Curva de aprendizagem guardada em results/learning_curve_maze.png")


# =====================================================
# 7. Guardar campeão
# =====================================================

ok, score = trainer.save_champion(
    "model/best_agent_maze.keras",
    env_factory=lambda: curriculum_env_factory(999999),
    max_steps=MAX_STEPS,
    n_eval=20,
    threshold=-2.0
)

print("Champion saved:", ok, "| score:", score)