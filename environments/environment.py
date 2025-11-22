# environment.py
from abc import ABC, abstractmethod
import numpy as np
from agents.agent import Agent

class Enviroment(ABC):

    def __init__(self, tamanho: tuple, dificuldade: int):
        self.tamanho = tamanho                 
        self.dificuldade = dificuldade
        self.mapa_estado = None                 
        self.agentes = {}
        self.posicoes_agentes = {}
        self.passo_tempo = 0

    @abstractmethod
    def criar_mapa(self, dificuldade) -> np.ndarray:
        pass

    @abstractmethod
    def atualizacao(self):
        """Chamado após todos os agentes agirem."""
        self.passo_tempo += 1

    @abstractmethod
    def agir(self, accao: int, agente: Agent):
        pass

    def regista_agente(self, agente: Agent, pos_inicial: tuple):
        self.agentes[agente.id] = agente
        self.posicoes_agentes[agente.id] = pos_inicial
        agente.posicao = tuple(pos_inicial)

    def get_posicao_agente(self, agente: Agent):
        return self.posicoes_agentes.get(agente.id)

    @abstractmethod
    def observacaoPara(self, agente: Agent):
        pass