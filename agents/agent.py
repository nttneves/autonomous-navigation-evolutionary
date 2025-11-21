# agent.py
from abc import ABC, abstractmethod
import queue
import json
import time

# TODO:PARA LER O AGENTE DO FICHEIRO NÃO PRECISO DE PARAMENTROS APENAS O NOME DO FICHEIRO 
class Agent(ABC):
    def __init__(self, id: str, politica=None, hasSensores=False):
        self.id = id
        self.politica = politica or None
        self.sensores = False
        self.last_observation = None
        self.last_action = None
        self.rewards = []
        self.ambiente = None
        self.posicao = {}

    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(data["id"], data.get("politica"), data.get("sensores"))

    @abstractmethod
    def observacao(self, obs): 
        pass

    @abstractmethod
    def age(self) -> int:
        pass

    @abstractmethod
    def avaliacaoEstadoAtual(self, recompensa: float):
        pass

    def instala(self, sensor):
        self.sensores = sensor