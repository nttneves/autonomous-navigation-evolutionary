# train_qlearning_maze.py
import json
import numpy as np
import matplotlib.pyplot as plt
from algorithms.qlearning_trainer import QLearningTrainer
from environments.environment_maze import MazeEnv
import random
import os

def make_env(dificuldade):
    return lambda: MazeEnv(dificuldade=dificuldade, max_steps=450)

curriculum_env_factories = [
    make_env(0),
    make_env(1),
    make_env(2),
]

def curriculum_env_factory(episode):
    if episode <= 1000:
        pool = curriculum_env_factories[0:2]
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

EPISODES = 10000
MAX_STEPS = 1000

print("=== TREINO Q-LEARNING - LABIRINTO ===\n")

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

os.makedirs("results/maze/train", exist_ok=True)
with open("results/maze/train/history_qlearning_maze.json", "w") as f:
    json.dump(history, f, indent=4)

print("\nHistórico guardado em results/maze/train/history_qlearning_maze.json")

if len(history) > 0:
    episodes = [h["episode"] for h in history]
    mean_rewards = [h["mean_reward"] for h in history]
    eval_rewards = [h["mean_eval_reward"] for h in history]
    best_scores = [h["best_score"] for h in history]
    epsilons = [h["epsilon"] for h in history]
    success_rates = [h["success_rate"] for h in history]
    success_rates_train = [h["success_rate_train"] for h in history]
    mean_steps = [h["mean_steps"] for h in history]
    q_table_sizes = [h["q_table_size"] for h in history]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.plot(episodes, mean_rewards, label="Mean Reward (Training)", alpha=0.7, linewidth=2)
    plt.plot(episodes, eval_rewards, label="Mean Reward (Evaluation)", linewidth=2)
    plt.plot(episodes, best_scores, label="Best Score", linewidth=2, linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning - Learning Curve (Rewards)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(episodes, epsilons, label="Epsilon (Exploration)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Decay")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(episodes, success_rates, label="Success Rate (Evaluation)", linewidth=2)
    plt.plot(episodes, success_rates_train, label="Success Rate (Training)", alpha=0.7, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.title("Goal Reaching Success Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 105])

    plt.subplot(2, 3, 4)
    plt.plot(episodes, mean_steps, label="Mean Steps", linewidth=2)
    plt.axhline(y=MAX_STEPS, color='r', linestyle='--', alpha=0.5, label=f'Max Steps ({MAX_STEPS})')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Mean Steps per Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(episodes, q_table_sizes, label="Q-Table Size", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Number of States")
    plt.title("Q-Table Growth (States Discovered)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 6)
    std_rewards = [h["std_reward"] for h in history]
    plt.plot(episodes, std_rewards, label="Reward Std Dev", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Standard Deviation")
    plt.title("Reward Variance (Learning Stability)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    os.makedirs("results/maze/train", exist_ok=True)
    plt.savefig("results/maze/train/plot_qlearning_maze.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos guardados em results/maze/train/plot_qlearning_maze.png")

    print("\n" + "="*60)
    print("ESTATÍSTICAS FINAIS")
    print("="*60)
    print(f"Episódios treinados: {EPISODES}")
    print(f"Melhor score: {trainer.best_score:.4f}")
    print(f"Último success rate (eval): {success_rates[-1]:.1f}%")
    print(f"Tamanho final da Q-table: {q_table_sizes[-1]} estados")
    print(f"Epsilon final: {epsilons[-1]:.4f}")
    print(f"Último reward médio (treino): {mean_rewards[-1]:.2f}")
    print(f"Último reward médio (eval): {eval_rewards[-1]:.2f}")
    print("="*60)

os.makedirs("model", exist_ok=True)
trainer.save_best_agent("model/best_agent_qlearning_maze")
print(f"\nMelhor agente guardado em model/best_agent_qlearning_maze.pkl")
print(f"Melhor score: {trainer.best_score:.4f}")
