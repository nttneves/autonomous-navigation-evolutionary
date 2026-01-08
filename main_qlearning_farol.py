import time
from environments.environment_farol import FarolEnv
from agents.qlearning_agent import QLearningAgent
from agents.QLearningRuntimeAgent import QLearningRuntimeAgent
from algorithms.qlearning_trainer import FarolObservationDiscretizer
from simulator.simulator import Simulator

# =====================================================
# MAIN - TESTE COM RENDER
# =====================================================
if __name__ == "__main__":
    # Carregar agente treinado
    path = "model/best_agent_qlearning_farol.pkl"
    q_agent = QLearningAgent.load(path)

    # Criar discretizer do treino
    discretizer = FarolObservationDiscretizer()

    # Criar runtime agent
    agent = QLearningRuntimeAgent(q_agent, discretizer, id="q_agent_1")

    # Criar ambiente
    env = FarolEnv(tamanho=(50,50), dificuldade=1, max_steps=300)

    # Criar simulator e adicionar agente
    sim = Simulator(env, max_steps=200)
    sim.agentes[agent.id] = agent

    # Reset inicial
    obs = sim.reset_env()

    # Criar renderer (janela Pygame)
    from environments.renderer import EnvRenderer
    renderer = EnvRenderer(env, window_size=500)

    done = False
    steps = 0

    while not done and steps < sim.max_steps:
        # ação do agente
        action = agent.age()

        # passo no simulador
        obs, reward, done, info = sim.step_once_for_visualiser(action)

        # desenhar agente no mapa
        alive = renderer.draw(env.posicoes_agentes)
        if not alive:  # se fechar a janela
            break

        steps += 1
        time.sleep(0.05)  # controla velocidade do boneco

    renderer.close()

    # resultado final
    final_pos = env.get_posicao_agente(agent)
    reached = final_pos == env.goal_pos
    print("\nRESULTADO FINAL")
    print(f"Reward total: {sim._total_reward:.2f}")
    print(f"Steps: {steps}")
    print(f"Posição final: {final_pos}")
    print(f"Goal: {env.goal_pos}")
    print(f"Chegou ao goal? {'✓' if reached else '✗'}")