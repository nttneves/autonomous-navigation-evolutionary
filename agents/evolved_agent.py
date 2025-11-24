# agents/evolved_agent.py
from agents.agent import Agent
import numpy as np
from algorithms.genetic import set_weights_vector
from model.model import create_rnn

class EvolvedAgent(Agent):
    def __init__(self, id: str, model=None, dim_input_rn: int = 10):
        super().__init__(id)
        self.politica = "evolved"
        if model is not None:
            self.rede_neuronal = model
        else:
            self.rede_neuronal = create_rnn(input_dim=dim_input_rn)
        self.genoma = None
        self.last_action = None

    @classmethod
    def cria(cls, ficheiro_json: str):
        import json
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(id=data["id"], dim_input_rn=data.get("dim_input", 10))

    def observacao(self, obs):
        self.last_observation = obs

    def set_genoma(self, genoma):
        self.genoma = genoma
        if genoma is not None:
            set_weights_vector(self.rede_neuronal, genoma)

    def _vetorizar_obs(self, obs):
        if obs is None:
            return np.zeros(10, dtype=np.float32)
        ranges = np.array(obs["ranges"], dtype=np.float32)   # 6
        radar = np.array(obs["radar"], dtype=np.float32)     # 4
        return np.concatenate([ranges, radar]).astype(np.float32)

    def age(self) -> int:
        """
        Produz ação discreta a partir de 2 outputs:
         - output[0] -> horizontal: >0 => right, <0 => left
         - output[1] -> vertical:  >0 => up,   <0 => down
        Para manter compatibilidade com o resto do código (4 ações), mapeamos:
          (dx_sign, dy_sign) -> action in {0..3}
        """
        vetor = self._vetorizar_obs(self.last_observation)
        out = self.rede_neuronal(np.array([vetor], dtype=np.float32), training=False)
        vals = out.numpy()[0]
        hx, hy = float(vals[0]), float(vals[1])

        # sinais
        if abs(hx) >= abs(hy):
            # mover horizontal
            if hx > 0:
                acao = 1  # right
            else:
                acao = 3  # left
        else:
            # mover vertical
            if hy > 0:
                acao = 0  # up (note: up is negative y in map coords, but we keep mapping consistent)
            else:
                acao = 2  # down

        self.last_action = acao
        return int(acao)

    def avaliacaoEstadoAtual(self, recompensa: float):
        self.regista_reward(recompensa)