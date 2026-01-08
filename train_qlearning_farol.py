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
def train_qlearning(agent, discretizer, episodes=2000, max_steps=200, verbose=True):
    rewards_history = []
    best_reward = -float('inf')
    best_agent_path = os.path.join('model', 'best_agent_qlearning_farol.pkl')
    os.makedirs('model', exist_ok=True)

    dificuldades = [1, 2, 3]  # 3 níveis de dificuldade
    pbar = trange(1, episodes + 1, desc="Treino Q-Learning")

    for ep in pbar:
        total_reward_ep = 0  # reward total do episódio

        for dificuldade in dificuldades:
            seed = np.random.randint(0, 10000)
            env = FarolEnv(tamanho=(50, 50), dificuldade=dificuldade, max_steps=max_steps, seed=seed)

            rl_agent = QLearningRuntimeAgent(id=f"q_agent_{ep}", discretizer=discretizer, agent=agent)
            env.regista_agente(rl_agent, pos_inicial=(10, 18))
            rl_agent.visited = set()  # para poder adicionar bónus exploração

            obs = env.observacaoPara(rl_agent)
            state = discretizer.tuple_to_index(discretizer.discretize(obs))

            for step in range(max_steps):
                action = agent.choose_action(state)
                prev_pos, new_pos, info = env.agir(action, rl_agent)

                # --- Reward do ambiente ---
                reward = -5.0
                done = False

                bx, by = env.goal_pos
                px, py = prev_pos
                nx, ny = new_pos

                
                delta = np.hypot(px - bx, py - by) - np.hypot(nx - bx, ny - by)

                if info.get("collision", False):
                    reward -= 15.0
                else:
                    reward += delta * (10.0 if delta > 0 else 50.0)

                if new_pos == env.goal_pos:
                    reward += 1000.0 + (max_steps - step) * 2.0
                    done = True


                env.atualizacao()
                obs2 = env.observacaoPara(rl_agent)
                state2 = discretizer.tuple_to_index(discretizer.discretize(obs2))

                agent.update(state, action, reward, state2, done)
                state = state2
                total_reward_ep += reward

                if done or env.terminou():
                    break

        # Decaimento lento de epsilon para manter exploração
        agent.decay_epsilon()
        rewards_history.append(total_reward_ep)

        # Guardar melhor agente
        if total_reward_ep > best_reward:
            best_reward = total_reward_ep
            agent.save(best_agent_path)

        if verbose:
            mean_recent = np.mean(rewards_history[-100:]) if len(rewards_history) >= 1 else 0
            pbar.set_postfix({"Reward": f"{total_reward_ep:.1f}", "Mean100": f"{mean_recent:.1f}", "ε": f"{agent.epsilon:.3f}"})

    return rewards_history, best_reward


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    EPISODES = 2000
    MAX_STEPS = 500

    print("=== TREINO Q-LEARNING - FAROL ===\n")


    discretizer = FarolObservationDiscretizer()
    agent = QLearningAgent(
        n_states=discretizer.n_states,
        n_actions=len(ACTION_TO_DELTA),
        alpha=0.1,
        gamma=0.99,
        epsilon=0.5,       # começa 100% explorando
        epsilon_min=0.3,   # nunca menos de 30% exploração
        epsilon_decay=0.999 # decaimento muito lento
    )

    rewards, best_reward = train_qlearning(agent, discretizer,
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