# random_agent.py
import numpy as np
from agent import Agent

class RandomAgent(Agent):
    def __init__(self, id: str, num_acoes: int):
        super().__init__(id)
        self.num_acoes = num_acoes

    @classmethod
    def cria(cls, ficheiro_json: str):
        import json
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(id=data["id"], num_acoes=data["num_acoes"])

    def observacao(self, obs):
        self.last_observation = obs

    def age(self) -> int:
        acao = np.random.randint(self.num_acoes)
        self.last_action = acao
        return acao

    def avaliacaoEstadoAtual(self, recompensa: float):
        self.regista_reward(recompensa)
