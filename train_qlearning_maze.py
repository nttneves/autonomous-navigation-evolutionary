# train_qlearning_maze.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from algorithms.qlearning_trainer import MazeObservationDiscretizer
from environments.environment_maze import MazeEnv
from environments.environment import ACTION_TO_DELTA
from agents.qlearning_agent import QLearningAgent

MAX_STEPS_BY_DIFF = {
    0: 50,     
    1: 150,  
    2: 600    
}


# ============================================================
# Treino Q-Learning (Maze – multi-dificuldade)
# ============================================================
def train_qlearning(agent, episodes=2000, verbose=True):
    rewards_history = []
    best_reward = -float("inf")

    os.makedirs("model", exist_ok=True)
    best_agent_path = "model/best_agent_qlearning_maze.pkl"

    dificuldades = [0, 1, 2]

    for ep in range(1, episodes + 1):
        total_reward_ep = 0.0

        for dificuldade in dificuldades:
            env = MazeEnv(dificuldade=dificuldade, max_steps=MAX_STEPS_BY_DIFF[dificuldade])

            # registar agente diretamente
            env.regista_agente(agent, pos_inicial=(1, 1))
            agent.reset()

            # observação inicial
            obs = env.observacaoPara(agent)
            agent.observacao(obs)

            visited = set()

            for step in range(MAX_STEPS_BY_DIFF[dificuldade]):
                action = agent.age()
                prev_pos, new_pos, info = env.agir(action, agent)

                bx, by = env.goal_pos
                px, py = prev_pos
                nx, ny = new_pos

                # ---------------- REWARD ----------------
                reward = -1.0
                done = False

                if info.get("collision", False):
                    reward -= 10.0
                else:
                    delta = np.hypot(px - bx, py - by) - np.hypot(nx - bx, ny - by)
                    reward += max(0.0, delta) * 7.0

                    if new_pos not in visited:
                        reward += 20.0
                        visited.add(new_pos)
                    else:
                        reward -= 4.0

                if new_pos == env.goal_pos:
                    reward += 1000.0
                    done = True
                # ----------------------------------------

                env.atualizacao()

                obs2 = env.observacaoPara(agent)
                agent.observacao(obs2)
                agent.avaliacaoEstadoAtual(reward, done)

                total_reward_ep += reward

                if done or env.terminou():
                    break

        agent.decay_epsilon()
        rewards_history.append(total_reward_ep)

        if total_reward_ep > best_reward:
            best_reward = total_reward_ep
            agent.save(best_agent_path)

        if verbose and ep % 100 == 0:
            mean_recent = np.mean(rewards_history[-100:])
            print(
                f"[Maze] Ep {ep} | "
                f"Reward={total_reward_ep:.1f} | "
                f"Mean100={mean_recent:.1f} | "
                f"ε={agent.epsilon:.3f}"
            )

    return rewards_history, best_reward


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    EPISODES = 10000

    print("=== TREINO Q-LEARNING - MAZE ===\n")

    discretizer = MazeObservationDiscretizer()

    agent = QLearningAgent(
        id="q_maze",
        discretizer=discretizer,
        n_states=discretizer.n_states,
        n_actions=len(ACTION_TO_DELTA),
        alpha=0.1,
        gamma=0.99,
        epsilon=0.7,
        epsilon_min=0.1,
        epsilon_decay=0.9998
    )

    rewards, best_reward = train_qlearning(
        agent,
        episodes=EPISODES,
        verbose=True
    )

    # Guardar histórico
    os.makedirs("results/maze/train", exist_ok=True)
    with open("results/maze/train/history_qlearning_maze.json", "w") as f:
        json.dump(
            {"rewards": rewards, "episodes": EPISODES, "best_reward": best_reward},
            f,
            indent=4
        )

    # Gráfico
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.xlabel("Episódio")
    plt.ylabel("Reward total")
    plt.title("Q-Learning – Maze")
    plt.grid(True)
    plt.savefig("results/maze/train/plot_qlearning_maze.png", dpi=200)
    plt.close()

    print("\nTREINO CONCLUÍDO")
    print(f"Melhor reward: {best_reward:.2f}")
    print(f"Epsilon final: {agent.epsilon:.4f}")