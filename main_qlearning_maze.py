# main_qlearning_maze.py
import time
from environments.environment_maze import MazeEnv
from agents.qlearning_agent import QLearningAgent
from algorithms.qlearning_trainer import MazeObservationDiscretizer
from simulator.simulator import Simulator

if __name__ == "__main__":
    discretizer = MazeObservationDiscretizer()

    agent = QLearningAgent.load(
        path="model/best_agent_qlearning_maze.pkl",
        discretizer=discretizer
    )

    dificuldade = 1
    env = MazeEnv(dificuldade=dificuldade, max_steps=400)

    sim = Simulator(env, max_steps=400)
    sim.agentes[agent.id] = agent

    sim.reset_env()

    from environments.renderer import EnvRenderer
    renderer = EnvRenderer(env, window_size=500)

    done = False
    steps = 0

    while not done and steps < sim.max_steps:
        action = agent.age()
        obs, reward, done, info = sim.step_once_for_visualiser(action)

        alive = renderer.draw(env.posicoes_agentes)
        if not alive:
            break

        steps += 1
        time.sleep(0.05)

    renderer.close()

    final_pos = env.get_posicao_agente(agent)
    print("\nRESULTADO FINAL")
    print(f"Reward total: {sim._total_reward:.2f}")
    print(f"Steps: {steps}")
    print(f"Posição final: {final_pos}")
    print(f"Goal: {env.goal_pos}")
    print(f"Chegou ao goal? {'✓' if final_pos == env.goal_pos else '✗'}")