"""
Funções de avaliação para uso pelo Algoritmo Genético
no processo evolutivo dos agentes com rede neuronal.
"""

# algorithms/evaluate.py
import numpy as np
from agents.evolved_agent import EvolvedAgent
from algorithms.genetic import set_weights_vector
from algorithms.train_evolution import _instalar_sensores


def evaluate_individual(weights, create_model, simulator):
    """
    Avalia um indivíduo da população genética.
    Retorna um Behaviour Characterization (BC).
    """

    # 1 — Criar modelo e carregar pesos
    model = create_model()
    set_weights_vector(model, weights)

    # 2 — Criar agente
    agent = EvolvedAgent(id="eval_agent", model=model)

    # 3 — Instalar sensores (ESSENCIAL!)
    _instalar_sensores(agent)

    # 4 — Executar episódio
    episode = simulator.run_episode(agent)

    # 5 — Usar posição final como BC (ou o que quiseres)
    bc = np.array(episode["final_pos"], dtype=float)

    return bc


def _instalar_sensores(agent):
    """
    Instala os sensores padrão num agente.
    O ambiente NÃO instala sensores.
    """
    from sensor.sensor_objeto import sensor_objeto
    from sensor.sensor_obstaculo import sensor_obstaculo

    s_obj = sensor_objeto(alcance=10)
    s_obj.dim = 8

    s_obs = sensor_obstaculo(alcance=10)
    s_obs.dim = 4

    agent.instala("sensor_objeto", s_obj)
    agent.instala("sensor_obstaculo", s_obs)