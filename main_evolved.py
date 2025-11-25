# main_evolved.py
import sys
import tensorflow as tf
from model.model import create_rnn
from simulator.simulator import Simulator
from environments.environment_farol import FarolEnv
from algorithms.trainer import EvolutionTrainer
from agents.evolved_agent import EvolvedAgent

# Exemplo simples: carregar modelo salvo e correr 1 episódio
def load_model_and_agent(path, input_dim=10):
    model = tf.keras.models.load_model(path)
    agent = EvolvedAgent(id="loaded", model=model)
    return agent

if __name__ == "__main__":
    path = "best_agent_farol.keras"
    try:
        agent = load_model_and_agent(path)
    except Exception as e:
        print("A carregar modelo:", e)
        sys.exit(1)

    print("Modelo carregado com sucesso!")

    env = FarolEnv(tamanho=(50,50), dificuldade=0, max_steps=200)
    sim = Simulator(env, max_steps=200)
    sim.agentes[agent.id] = agent

    # posição inicial e registo manual (sim.run_episode assume agentes registados)
    env.reset()
    start = (0, env.tamanho[1] - 1)
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