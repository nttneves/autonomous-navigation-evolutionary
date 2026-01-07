import sys
import numpy as np

from simulator.simulator import Simulator
from environments.environment_maze import MazeEnv
from agents.evolved_agent import EvolvedAgent
from model.model import create_mlp
from algorithms.genetic import set_weights_vector

# ============================================
# Carregar modelo salvo (.npy com vetor de pesos)
# ============================================
def load_model_and_agent(path, input_dim=12, hidden_units=32, outputs=4):
    genome = np.load(path)
    model = create_mlp(input_dim=input_dim, hidden_units=hidden_units, outputs=outputs)
    set_weights_vector(model, genome)
    agent = EvolvedAgent(id="loaded", model=model, dim_input_rn=input_dim)
    return agent

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    path = "model/best_agent_maze.npy"

    env = MazeEnv(dificuldade=2, max_steps=300)

    agent = load_model_and_agent(path)

    sim = Simulator(env, max_steps=300)
    sim.agentes[agent.id] = agent

    env.reset()
    start = (1, env.tamanho[1] - 1)
    env.regista_agente(agent, start)
    obs = env.observacaoPara(agent)
    agent.observacao(obs)

    res = sim.run_episode(render=True)

    print("===== RESULTADO FINAL =====")
    print("Total Reward:", res["total_reward"])
    print("Steps:", res["steps"])
    print("Reached Goal:", res["reached_goal"])
    print("Final Position:", res["final_pos"])
    print("Goal Position: ", res["goal_pos"])
    print("===========================")