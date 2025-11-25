# fixed_policy_agent.py
import numpy as np
from agents.agent import Agent

class FixedPolicyAgent(Agent):
    """
    Agente com política fixa baseada nos sensores:
      - Radar diz a direção do alvo (front/right/back/left)
      - Rangefinders dizem se está livre
      - Política: tenta ir na direção do alvo; se bloqueado, tenta virar.
    """

    def __init__(self, id: str):
        super().__init__(id)
        self.politica = "fixed"
        self.last_observation = None
        self.last_action = None

    @classmethod
    def cria(cls, ficheiro_json: str):
        import json
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(id=data["id"])

    def observacao(self, obs):
        self.last_observation = obs

    def _direcao_alvo(self):
        radar = self.last_observation["radar"]  # [front, right, back, left]
        # retorna índice da direção dominante
        return int(np.argmax(radar))   # 0=front,1=right,2=back,3=left

    def _livre(self, direcao):
        """
        rangefinders (obs["ranges"]):
            0=up,1=right,2=down,3=left,4=up-right,5=up-left
        Para política principal só usamos 0..3.
        range > 0 significa espaço livre até 10 passos de distância.
        """
        ranges = self.last_observation["ranges"]
        return ranges[direcao] > 0.0

    def age(self) -> int:
        """
        0=up, 1=right, 2=down, 3=left
        Política:
           1. tenta direção do alvo
           2. senão tenta virar à direita
           3. senão tenta virar à esquerda
           4. para baixo
           5.fallback: tenta inversa
        """
        if self.last_observation is None:
            self.last_action = 0
            return 0

        # 1 — direção desejada (0..3)
        alvo = self._direcao_alvo()

        # mapa para ações compatíveis:
        # radar: front=0 (up), right=1, back=2 (down), left=3
        desired = alvo

        ranges = self.last_observation["ranges"]
        if self.last_action is not None and ranges[self.last_action] > 0.0:
            return self.last_action
        # 1) tentar direção principal
        if self._livre(desired):
            self.last_action = desired
            return desired
       
        
        
        
        # 2) tentar virar à direita
        direita = (desired + 1) % 4
        if self._livre(direita):
            self.last_action = direita
            return direita

        # 3) tentar virar à esquerda
        esquerda = (desired - 1) % 4
        if self._livre(esquerda):
            self.last_action = esquerda
            return esquerda
        

        # 4) fallback: inversa
        inverso = (desired + 2) % 4
        self.last_action = inverso
        return inverso
    

    def avaliacaoEstadoAtual(self, recompensa: float):
        self.regista_reward(recompensa)