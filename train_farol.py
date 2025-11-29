# train_farol.py
from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
from environments.environment_farol import FarolEnv
import json
import matplotlib.pyplot as plt


def make_env():
    return FarolEnv(tamanho=(50,50), dificuldade=3, max_steps=300)

# criar trainer
trainer = EvolutionTrainer(
    model_builder=lambda: create_mlp(input_dim=10),
    pop_size=80,
    archive_prob=0.12,
    elite_fraction=0.05
)

# treinar
history = trainer.train(
    env_factory=make_env,
    max_steps=200,
    generations=20,
    episodes_per_individual=3,
    alpha=0.7,
    verbose=True
)

# --- guardar histórico ---
with open("results/history_farol.json", "w") as f:
    json.dump(history, f, indent=4)
print("Histórico guardado em results/history_farol.json")

# --- curva de aprendizagem ---
gens = [h["generation"] for h in history]
mean_fit = [h["mean_fitness"] for h in history]
max_fit = [h["max_fitness"] for h in history]

plt.figure(figsize=(10,5))
plt.plot(gens, mean_fit, label="Mean Fitness", linewidth=2)
plt.plot(gens, max_fit, label="Max Fitness", linewidth=2)
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Curva de Aprendizagem – Ambiente Farol")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curve_farol.png", dpi=300)
plt.close()

print("Curva de aprendizagem guardada em results/learning_curve_farol.png")

# --- guardar champion ---
ok, score = trainer.save_champion(
    "model/best_agent_farol.keras",
    make_env,
    max_steps=200,
    n_eval=12,
    threshold=-0.5
)
print("Champion saved:", ok, "| score:", score)