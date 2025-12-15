# agent.py
from abc import ABC, abstractmethod
import json
import numpy as np

class Agent(ABC):

    def __init__(self, id: str, politica: str=None, sensores: bool=True):
        self.id = id
        self.politica = politica
        self.sensores = sensores

        self.sensors_range = 5 if sensores else 1

        self.last_observation = None
        self.last_action = None
        self.rewards = []
        self.posicao = None

    # ---------------------------------------------------------
    # Factory a partir de ficheiro JSON
    # ---------------------------------------------------------
    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)

        return cls(
            id=data["id"],
            sensores=data.get("sensores", True)
        )

    # ---------------------------------------------------------
    # Métodos utilitários comuns a todos os agentes
    # ---------------------------------------------------------
    def observacao(self, obs):
        """Recebe observação do ambiente (ranges + radar)."""
        self.last_observation = obs

    def regista_reward(self, r: float):
        self.rewards.append(r)

    def avaliacaoEstadoAtual(self, recompensa: float):
        """Todos usam esta versão, mas subclasses podem estender."""
        self.regista_reward(recompensa)
    
    def instala(self, sensor: bool):
        """Associa sensores ao agente."""
        self.sensores = sensor

    # ---------------------------------------------------------
    @abstractmethod
    def age(self) -> int:
        """Escolhe ação com base no estado interno."""
        pass