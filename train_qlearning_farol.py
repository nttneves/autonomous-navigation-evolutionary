# train_qlearning_farol.py
import json
import numpy as np
import matplotlib.pyplot as plt
from algorithms.qlearning_trainer import QLearningTrainer
from environments.environment_farol import FarolEnv
import random
import os

def make_env(dificuldade, seed):
    return lambda: FarolEnv(
        tamanho=(50, 50),
        dificuldade=dificuldade,
        max_steps=350,
        seed=seed
    )

curriculum_env_factories = []

for s in range(100, 110):
    curriculum_env_factories.append(make_env(dificuldade=1, seed=s))

for s in range(200, 210):
    curriculum_env_factories.append(make_env(dificuldade=2, seed=s))

for s in range(300, 310):
    curriculum_env_factories.append(make_env(dificuldade=3, seed=s))

random.shuffle(curriculum_env_factories)

def curriculum_env_factory(episode):
    if episode <= 1000:
        pool = curriculum_env_factories[0:20]
    else:
        pool = curriculum_env_factories
    return random.choice(pool)()

trainer = QLearningTrainer(
    input_dim=12,
    learning_rate=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9995,
    n_bins=8,
    seed=42,
    decay_per_step=False,
    min_explore_steps=1000,
    reward_clip=None,
    normalize_rewards=False,
    verbose=True
)

EPISODES = 2000
MAX_STEPS = 350

print("=== TREINO Q-LEARNING - FAROL ===\n")

class EnvFactoryWrapper:
    def __init__(self, curriculum_func):
        self.curriculum_func = curriculum_func
        self.episode_count = 0
    
    def __call__(self):
        self.episode_count += 1
        return self.curriculum_func(self.episode_count)

env_factory_wrapper = EnvFactoryWrapper(curriculum_env_factory)

history, trained_agent = trainer.train(
    env_factory=env_factory_wrapper,
    max_steps=MAX_STEPS,
    episodes=EPISODES,
    eval_frequency=100,
    eval_episodes=10,
    verbose=True
)

os.makedirs("results/farol/train", exist_ok=True)
with open("results/farol/train/history_qlearning_farol.json", "w") as f:
    json.dump(history, f, indent=4)

print("\nHistórico guardado em results/farol/train/history_qlearning_farol.json")

if len(history) > 0:
    episodes = [h["episode"] for h in history]
    mean_rewards = [h["mean_reward"] for h in history]
    eval_rewards = [h["mean_eval_reward"] for h in history]
    best_scores = [h["best_score"] for h in history]
    best_success_rates = [h.get("best_success_rate", 0.0) for h in history]
    epsilons = [h["epsilon"] for h in history]
    success_rates = [h["success_rate"] for h in history]
    success_rates_train = [h["success_rate_train"] for h in history]
    mean_steps = [h["mean_steps"] for h in history]
    q_table_sizes = [h["q_table_size"] for h in history]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(episodes, mean_rewards, label="Mean Reward (Training)", alpha=0.7, linewidth=2, color="blue")
    plt.plot(episodes, eval_rewards, label="Mean Reward (Evaluation)", linewidth=2, color="green")
    plt.plot(episodes, best_scores, label="Best Score", linewidth=2, linestyle="--", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning - Learning Curve (Rewards)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(episodes, epsilons, label="Epsilon (Exploration)", linewidth=2, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Decay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(episodes, success_rates, label="Success Rate (Evaluation)", linewidth=2, color="green")
    plt.plot(episodes, success_rates_train, label="Success Rate (Training)", alpha=0.7, linewidth=2, color="blue")
    plt.plot(episodes, best_success_rates, label="Best Success Rate", linewidth=2, linestyle="--", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.title("Goal Reaching Success Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 105])
    
    plt.subplot(2, 3, 4)
    plt.plot(episodes, mean_steps, label="Mean Steps", linewidth=2, color="purple")
    plt.axhline(y=MAX_STEPS, color='r', linestyle='--', alpha=0.5, label=f'Max Steps ({MAX_STEPS})')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Mean Steps per Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(episodes, q_table_sizes, label="Q-Table Size", linewidth=2, color="brown")
    plt.xlabel("Episode")
    plt.ylabel("Number of States")
    plt.title("Q-Table Growth (States Discovered)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 3, 6)
    std_rewards = [h["std_reward"] for h in history]
    plt.plot(episodes, std_rewards, label="Reward Std Dev", linewidth=2, color="crimson")
    plt.xlabel("Episode")
    plt.ylabel("Standard Deviation")
    plt.title("Reward Variance (Learning Stability)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/farol/train/plot_qlearning_farol.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráficos guardados em results/farol/train/plot_qlearning_farol.png")
    
    print("\n" + "="*60)
    print("ESTATÍSTICAS FINAIS")
    print("="*60)
    print(f"Episódios treinados: {EPISODES}")
    print(f"Melhor score: {trainer.best_score:.4f}")
    print(f"Melhor success rate: {trainer.best_success_rate:.1f}%")
    print(f"Último success rate (eval): {success_rates[-1]:.1f}%")
    print(f"Tamanho final da Q-table: {q_table_sizes[-1]} estados")
    print(f"Epsilon final: {epsilons[-1]:.4f}")
    print(f"Último reward médio (treino): {mean_rewards[-1]:.2f}")
    print(f"Último reward médio (eval): {eval_rewards[-1]:.2f}")
    print("="*60)

os.makedirs("model", exist_ok=True)
trainer.save_best_agent("model/best_agent_qlearning_farol")
print(f"\nMelhor agente guardado em model/best_agent_qlearning_farol.pkl")
print(f"Melhor score: {trainer.best_score:.4f}")
print(f"Melhor success rate: {trainer.best_success_rate:.1f}%")
