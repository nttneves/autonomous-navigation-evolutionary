# main_test_agent.py
import numpy as np
import tensorflow as tf

from environments.environment_farol import FarolEnv
from agents.evolved_agent import EvolvedAgent
from model.model import create_mlp
from simulator.simulator import Simulator

def carregar_agente(model_path: str):
    """
    Carrega o modelo .keras e devolve um EvolvedAgent totalmente funcional.
    """
    print(f"A carregar modelo: {model_path}")

    # carregar modelo treinado
    model = tf.keras.models.load_model(model_path)

    # extrair pesos como genoma
    pes = model.get_weights()
    flat = np.concatenate([p.flatten() for p in pes])

    # criar agente
    agent = EvolvedAgent(id="champion", model=model)
    agent.set_genoma(flat)

    print("Modelo carregado com sucesso!")
    return agent

def main():
    # ===============================
    # 1) Carregar agente treinado
    # ===============================
    agent = carregar_agente("best_agent_model.keras")

    # ===============================
    # 2) Criar ambiente
    # ===============================
    env = FarolEnv(
        tamanho=(21,21),
        dificuldade=0,
        max_steps=200
    )
    
    # ===============================
    # 3) Criar simulador
    # ===============================
    sim = Simulator(env, max_steps=200)
    sim.agentes[agent.id] = agent

    # ===============================
    # 4) Correr episódio com GUI
    # ===============================
    print("\nA correr episódio com renderização...")
    resultado = sim.run_episode(render=True)

    print("\n===== RESULTADO FINAL =====")
    print(f"Total Reward: {resultado['total_reward']}")
    print(f"Steps: {resultado['steps']}")
    print(f"Reached Goal: {resultado['reached_goal']}")
    print(f"Final Position: {resultado['final_pos']}")
    print(f"Goal Position:  {resultado['goal_pos']}")
    print("===========================")

if __name__ == "__main__":
    main()