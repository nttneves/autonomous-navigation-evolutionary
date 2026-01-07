# train_qlearning_farol.py
import json
import numpy as np
import matplotlib.pyplot as plt
from algorithms.qlearning_trainer import FarolObservationDiscretizer
from environments.environment_farol import FarolEnv
from environments.environment import ACTION_TO_DELTA
from agents.qlearning_agent import QLearningAgent
from agents.QLearningRuntimeAgent import QLearningRuntimeAgent
import os
from tqdm import trange  # Barra de progresso

# ============================================================
# Treino Q-Learning
# ============================================================
def train_qlearning(env, agent, discretizer, episodes=2000, max_steps=200, verbose=True):
    rewards_history = []
    best_reward = -float('inf')
    best_agent_path = os.path.join('model', 'best_agent_qlearning_farol.pkl')
    os.makedirs('model', exist_ok=True)

    pbar = trange(1, episodes + 1, desc="Treino Q-Learning")

    for ep in pbar:
        # Reset do ambiente
        env.reset()
        rl_agent = QLearningRuntimeAgent(id=f"q_agent_{ep}", discretizer=discretizer, agent=agent)
        env.regista_agente(rl_agent, pos_inicial=(10, 18))

        obs = env.observacaoPara(rl_agent)
        state = discretizer.tuple_to_index(discretizer.discretize(obs))

        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)

            prev_pos, new_pos, info = env.agir(action, rl_agent)
            reward, done = env.compute_reward(rl_agent, prev_pos, new_pos, info)
            env.atualizacao()

            obs2 = env.observacaoPara(rl_agent)
            state2 = discretizer.tuple_to_index(discretizer.discretize(obs2))

            agent.update(state, action, reward, state2, done)

            state = state2
            total_reward += reward

            if done or env.terminou():
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        # Guardar melhor agente
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_agent_path)

        # Atualizar TUI da barra de progresso
        if verbose:
            mean_recent = np.mean(rewards_history[-100:]) if len(rewards_history) >= 1 else 0
            pbar.set_postfix(
                {"Reward": f"{total_reward:.1f}", "Mean100": f"{mean_recent:.1f}", "ε": f"{agent.epsilon:.3f}"}
            )

    return rewards_history, best_reward

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    EPISODES = 2000
    MAX_STEPS = 400

    print("=== TREINO Q-LEARNING - FAROL ===\n")

    env = FarolEnv(tamanho=(50, 50), dificuldade=2, max_steps=MAX_STEPS, seed=42)
    discretizer = FarolObservationDiscretizer()
    agent = QLearningAgent(
        n_states=discretizer.n_states,
        n_actions=len(ACTION_TO_DELTA),
        alpha=0.1,
        gamma=0.99
    )

    rewards, best_reward = train_qlearning(env, agent, discretizer,
                                           episodes=EPISODES, max_steps=MAX_STEPS, verbose=True)

    # ============================================================
    # Salvar histórico
    # ============================================================
    os.makedirs("results/farol/train", exist_ok=True)
    history_path = "results/farol/train/history_qlearning_farol.json"
    with open(history_path, "w") as f:
        json.dump({"rewards": rewards, "episodes": EPISODES, "best_reward": best_reward}, f, indent=4)
    print(f"\nHistórico guardado em {history_path}")

    # ============================================================
    # Gráficos
    # ============================================================
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(rewards)+1), rewards, label="Reward por episódio")
    plt.xlabel("Episódio")
    plt.ylabel("Reward total")
    plt.title("Q-Learning - Treino Farol")
    plt.grid(True)
    plt.legend()
    plot_path = "results/farol/train/plot_qlearning_farol.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado em {plot_path}")

    # ============================================================
    # Estatísticas finais
    # ============================================================
    print("\n" + "="*50)
    print("TREINO CONCLUÍDO - ESTATÍSTICAS FINAIS")
    print("="*50)
    print(f"Episódios treinados: {EPISODES}")
    print(f"Melhor reward: {best_reward:.2f}")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    print(f"Tamanho final da Q-table: {agent.Q.shape[0]} estados")
    print(f"Último reward: {rewards[-1]:.2f}")
    print("="*50)

    # ============================================================
    # Garantir melhor agente salvo
    # ============================================================
    os.makedirs("model", exist_ok=True)
    best_agent_path = "model/best_agent_qlearning_farol.pkl"
    if not os.path.exists(best_agent_path):
        agent.save(best_agent_path)
    print(f"\nMelhor agente guardado em {best_agent_path}")