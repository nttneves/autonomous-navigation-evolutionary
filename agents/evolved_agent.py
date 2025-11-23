# evolved_agent.py
from agents.agent import Agent
from model.model import create_mlp
import numpy as np
from algorithms.genetic import set_weights_vector

class EvolvedAgent(Agent):

  def __init__(self, id: str, model=None):
        super().__init__(id)
        self.input_dim = 10

        if model is not None:
            self.rede_neuronal = model
        else:
            self.rede_neuronal = create_mlp(input_dim= self.input_dim)

        self.genoma = None
        self.last_action = None

  @classmethod
  def cria(cls, ficheiro_json: str):
    import json
    with open(ficheiro_json, "r") as f:
      data = json.load(f)

    return cls(id=data["id"])
  
  def observacao(self, obs):
    """Compatível com o método abstrato da classe base."""
    self.last_observation = obs
    
  def set_genoma(self, genoma):
    """Atualizar pesos da rede a partir de vetor plano."""
    self.genoma = genoma
    set_weights_vector(self.rede_neuronal, genoma)

  def _vetorizar_obs(self, obs):
    if obs is None:
        return np.zeros(10, dtype=np.float32)

    sensores = np.array(obs["sensores"], dtype=np.float32)     # 8 valores
    dir_farol = np.array(obs["dir_farol"], dtype=np.float32)   # 2 valores

    return np.concatenate([sensores, dir_farol])
  
  def age(self):
    """
    Calcula a ação baseada na rede neuronal.
    O simulador chama: a = agent.age(obs)
    """
    vetor = self._vetorizar_obs(self.last_observation)

    # out = self.rede_neuronal(
    #   np.array([vetor], dtype=np.float32),
    #   training=False
    # )

    # RNN precisa de (batch=1, timesteps=1, features=input_dim)
    inp = np.array(vetor, dtype=np.float32).reshape(1, 1, -1)

    out = self.rede_neuronal(inp, training=False)

    acao = int(np.argmax(out.numpy()))
    self.last_action = acao
    return acao

  def avaliacaoEstadoAtual(self, recompensa: float):
    """Regista reward (interface exigida pelo enunciado)."""
    self.regista_reward(recompensa)