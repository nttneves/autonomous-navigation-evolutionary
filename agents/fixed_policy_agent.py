# fixed_policy_agent.py
import numpy as np
from agents.agent import Agent

class FixedPolicyAgent(Agent):
    """
    Política fixa híbrida:
      - Se paredes estão próximas -> segue parede (right-hand rule)
      - Se não há paredes -> usa radar para aproximar-se do alvo
    """

    def __init__(self, id: str):
        super().__init__(id, politica="fixed")
        self.last_observation = None
        self.last_action = 0

    @classmethod
    def cria(cls, ficheiro_json: str):
        import json
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(id=data["id"])

    def observacao(self, obs):
        self.last_observation = obs

    def livre(self, d):
        """range > 0 -> espaço livre"""
        return self.last_observation["ranges"][d] > 0.0

    def paredes_proximas(self):
        """Se algum rangefinder cardinal (up/right/down/left) < 1 → há paredes próximas."""
        ranges = self.last_observation["ranges"]
        return min(ranges[0:4]) < 1.0

    def right_hand_rule(self):
        """Implementa wall-following (follow right wall)."""
        d = self.last_action
        direita = (d + 1) % 4
        esquerda = (d - 1) % 4
        tras = (d + 2) % 4

        # 1) tenta virar à direita
        if self.livre(direita):
            return direita

        # 2) se não, tenta frente
        if self.livre(d):
            return d

        # 3) se não, tenta esquerda
        if self.livre(esquerda):
            return esquerda

        # 4) fallback: trás
        return tras

    def radar_direction(self):
        """Front=0, right=1, back=2, left=3."""
        return int(np.argmax(self.last_observation["radar"]))

    def age(self) -> int:
        if self.last_observation is None:
            self.last_action = 0
            return 0

        ranges = self.last_observation["ranges"]

        # Número de paredes próximas em direções cardinais
        num_paredes = sum(1 for v in ranges[0:4] if v < 1.0)

        # ---------------------------------------------------
        # 1) Corredor / zona apertada → wall-following
        # ---------------------------------------------------
        if sum(1 for v in ranges if v < 1.0) >= 4:
            a = self.right_hand_rule()
            self.last_action = a
            return a

        # ---------------------------------------------------
        # 2) Espaço aberto → segue radar
        # ---------------------------------------------------
        desired = self.radar_direction()

        # Se a direção do radar está livre → usa
        if self.livre(desired):
            self.last_action = desired
            return desired

        # Senão → fallback para wall-following
        a = self.right_hand_rule()
        self.last_action = a
        return a

    def avaliacaoEstadoAtual(self, recompensa: float):
        self.regista_reward(recompensa)