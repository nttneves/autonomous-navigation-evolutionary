import json
import numpy as np
import matplotlib.pyplot as plt
from algorithms.qlearning_trainer import MazeObservationDiscretizer
from environments.environment_maze import MazeEnv
from environments.environment import ACTION_TO_DELTA
from agents.qlearning_agent import QLearningAgent
from agents.QLearningRuntimeAgent import QLearningRuntimeAgent
import os
from tqdm import trange

# ============================================================
# Treino Q-Learning (Maze – multi-mapas) com incentivo a explorar
# ============================================================
def train_qlearning(agent, discretizer, episodes=2000, max_steps=200, verbose=True):
    rewards_history = []
    best_reward = -float('inf')

    os.makedirs("model", exist_ok=True)
    best_agent_path = "model/best_agent_qlearning_maze.pkl"

    dificuldades = [0, 1, 2]  # os 3 mazes
    pbar = trange(1, episodes + 1, desc="Treino Q-Learning (Maze)")

    for ep in pbar:
        total_reward_ep = 0  # reward total do episódio

        for dificuldade in dificuldades:  # cada episódio passa pelos 3 níveis
            env = MazeEnv(dificuldade=dificuldade, max_steps=max_steps)

            rl_agent = QLearningRuntimeAgent(
                id=f"q_agent_{ep}",
                discretizer=discretizer,
                agent=agent
            )
            env.regista_agente(rl_agent, pos_inicial=(1, 1))
            rl_agent.visited = set()

            obs = env.observacaoPara(rl_agent)
            state = discretizer.tuple_to_index(discretizer.discretize(obs))

            for step in range(max_steps):
                action = agent.choose_action(state)
                prev_pos, new_pos, info = env.agir(action, rl_agent)
                bx, by = env.goal_pos
                px, py = prev_pos
                nx, ny = new_pos

                # Reward para incentivar exploração
                reward = -1
                done = False

                if info.get("collision", False):
                    reward -= 10.0
                else:
                    delta = max(0, np.hypot(px - bx, py - by) - np.hypot(nx - bx, ny - by))
                    reward += delta * 7.0

                    if (nx, ny) not in rl_agent.visited:
                        reward += 20.0
                        rl_agent.visited.add((nx, ny))
                    else:
                        reward -= 4.0

                if (nx, ny) == (bx, by):
                    reward += 1000.0
                    done = True

                env.atualizacao()
                obs2 = env.observacaoPara(rl_agent)
                state2 = discretizer.tuple_to_index(discretizer.discretize(obs2))

                agent.update(state, action, reward, state2, done)
                state = state2
                total_reward_ep += reward

                if done or env.terminou():
                    break

        # Decaimento lento de epsilon
        agent.decay_epsilon()
        rewards_history.append(total_reward_ep)

        # Guardar melhor agente
        if total_reward_ep > best_reward:
            best_reward = total_reward_ep
            agent.save(best_agent_path)

        if verbose:
            mean_recent = np.mean(rewards_history[-100:])
            pbar.set_postfix({
                "R": f"{total_reward_ep:.1f}",
                "Mean100": f"{mean_recent:.1f}",
                "ε": f"{agent.epsilon:.3f}"
            })

    return rewards_history, best_reward


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    EPISODES = 10000
    MAX_STEPS = 600

    print("=== TREINO Q-LEARNING - MAZE ===\n")

    discretizer = MazeObservationDiscretizer()
    agent = QLearningAgent(
        n_states=discretizer.n_states,
        n_actions=len(ACTION_TO_DELTA),
        alpha=0.1,           # taxa de aprendizagem média
        gamma=0.99,          # valor futuro muito importante
        epsilon=0.7,         # começa 100% explorando
        epsilon_min=0.1,     # nunca menos de 30% exploração
        epsilon_decay=0.9998 # decaimento muito lento
    )
    rewards, best_reward = train_qlearning(
        agent,
        discretizer,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        verbose=True
    )

    # Guardar histórico
    os.makedirs("results/maze/train", exist_ok=True)
    history_path = "results/maze/train/history_qlearning_maze.json"
    with open(history_path, "w") as f:
        json.dump({"rewards": rewards, "episodes": EPISODES, "best_reward": best_reward}, f, indent=4)
    print(f"\nHistórico guardado em {history_path}")

    # Gráfico
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Reward por episódio")
    plt.xlabel("Episódio")
    plt.ylabel("Reward total")
    plt.title("Q-Learning – Treino Multi-Labirinto (Maze)")
    plt.grid(True)
    plt.legend()
    plot_path = "results/maze/train/plot_qlearning_maze.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em {plot_path}")

    # Estatísticas finais
    print("\n" + "=" * 50)
    print("TREINO CONCLUÍDO - ESTATÍSTICAS FINAIS")
    print("=" * 50)
    print(f"Episódios treinados: {EPISODES}")
    print(f"Melhor reward: {best_reward:.2f}")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    print(f"Tamanho da Q-table: {agent.Q.shape[0]} estados")
    print(f"Último reward: {rewards[-1]:.2f}")
    print("=" * 50)
    print(f"\nMelhor agente guardado em model/best_agent_qlearning_maze.pkl")