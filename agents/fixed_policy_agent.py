# agents/fixed_policy_agent.py
import numpy as np
from agents.agent import Agent

class FixedPolicyAgent(Agent):

    def __init__(self, id: str, sensores: bool=True):
        super().__init__(id, politica="fixed", sensores=sensores)
        self.last_action = 0

    @classmethod
    def cria(cls, ficheiro_json: str):
        import json

        with open(ficheiro_json, "r") as f:
            data = json.load(f)

        agent_id = data["id"]
        sensores = bool(data.get("sensores", True))

        return cls(
            id=agent_id,
            sensores=sensores)

    # ---------------------------------------------------------
    def livre(self, d):
        return self.last_observation["ranges"][d] > 0.0

    def right_hand_rule(self):
        d = self.last_action
        direita = (d + 1) % 4
        esquerda = (d - 1) % 4
        tras = (d + 2) % 4

        if self.livre(direita): return direita
        if self.livre(d):       return d
        if self.livre(esquerda): return esquerda
        return tras

    def radar_direction(self):
        return int(np.argmax(self.last_observation["radar"]))

    # ---------------------------------------------------------
    def age(self) -> int:
        if self.last_observation is None:
            return 0

        ranges = self.last_observation["ranges"]

        # Se muitos sensores detetam paredes, segue parede
        if sum(1 for v in ranges[:4] if v < 1.0) >= 3:
            a = self.right_hand_rule()
            self.last_action = a
            return a

        # Espaço aberto → radar
        desired = self.radar_direction()
        if self.livre(desired):
            self.last_action = desired
            return desired

        a = self.right_hand_rule()
        self.last_action = a
        return a