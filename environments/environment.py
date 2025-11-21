from abc import ABC, abstractmethod
import numpy as np
from agents.agent import Agent

class Enviroment(ABC):
  
  def __init__(self, tamanho: tuple, dificulade: int):
    self.tamanho = tamanho
    self.mapa_estado = self.criar_mapa(dificulade)
    self.agentes = {}
    self.posicoes_agentes = {}
    self.passo_tempo = 0
    
    
  @abstractmethod
  def criar_mapa(self, dificuldade) -> np.ndarray:
    pass
    
    
  @abstractmethod
  def atualizacao(self):
    """Chamado APÓS todos os agentes agirem."""
    self.passo_tempo += 1
    pass

  @abstractmethod
  def agir(self, accao: str, agente):
    pass
    
  def regista_agente(self, agente: Agent, pos_inicial: tuple):
    self.agentes[agente.id] = agente
    self.posicoes_agentes[agente.id] = pos_inicial
    try:
        agente.posicao = tuple(pos_inicial)
    except Exception:
        pass

  def get_posicao_agente(self, agente):
    return self.posicoes_agentes.get(agente.id)

  @abstractmethod
  def observacaoPara(self, agente):
    pass
    