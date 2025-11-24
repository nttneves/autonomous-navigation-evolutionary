# agent.py
from abc import ABC, abstractmethod
import json

# TODO:PARA LER O AGENTE DO FICHEIRO NÃO PRECISO DE PARAMENTROS APENAS O NOME DO FICHEIRO 
class Agent(ABC):

    def __init__(self, id: str, politica: str=None, hasSensores=True):
        self.id = id
        self.politica = politica
        self.sensores = bool(hasSensores)
        self.last_observation = None
        self.last_action = None
        self.rewards = []
        self.ambiente = None
        self.posicao = None

    @classmethod
    def cria(cls, ficheiro_json: str):
        """Cria o agente a partir do ficheiro de parâmetros."""
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(
            id=data["id"],
            politica=data.get("politica"),
            hasSensores=data.get("sensores", False)
        )

    @abstractmethod
    def observacao(self, obs):
        """Recebe observação do ambiente."""
        pass

    @abstractmethod
    def age(self) -> int:
        """Escolhe a ação com base no estado interno."""
        pass

    @abstractmethod
    def avaliacaoEstadoAtual(self, recompensa: float):
        """Recebe feedback/reward do ambiente."""
        pass

    def instala(self, sensor: bool):
        """Associa sensores ao agente."""
        self.sensores = sensor

    def regista_reward(self, r: float):
        self.rewards.append(r)