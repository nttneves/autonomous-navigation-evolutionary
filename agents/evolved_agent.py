# agents/evolved_agent.py
from agents.agent import Agent
import numpy as np
from algorithms.genetic import set_weights_vector
from model.model import create_mlp
import json
import tensorflow as tf

class EvolvedAgent(Agent):

    def __init__(self, id: str, model=None, dim_input_rn: int = 10, sensores: bool=True):
        super().__init__(id, politica="evolved", sensores=sensores)

        self.dim_input_rn = dim_input_rn
        self.rede_neuronal = model if model is not None else create_mlp(input_dim=dim_input_rn)
        self.genoma = None

    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)

        agent_id = data["id"]
        dim_input = data.get("dim_input", 10)

        model_path = data.get("path")
        model = tf.keras.models.load_model(model_path)

        sensores = bool(data["sensores"])

        ag = cls(
            id=agent_id,
            model=model,
            dim_input_rn=dim_input,
            sensores=sensores)

        return ag

    def set_genoma(self, genoma):
        self.genoma = genoma
        if genoma is not None:
            set_weights_vector(self.rede_neuronal, genoma)

    # ---------------------------------------------------------
    def vetorizar_obs(self, obs):
        if obs is None:
            return np.zeros(self.dim_input_rn)

        ranges = np.array(obs["ranges"], dtype=np.float32)
        radar = np.array(obs["radar"], dtype=np.float32)
        return np.concatenate([ranges, radar], dtype=np.float32)

    # ---------------------------------------------------------
    def age(self) -> int:
        vetor = self.vetorizar_obs(self.last_observation)
        out = self.rede_neuronal(np.array([vetor], dtype=np.float32), training=False).numpy()[0]

        hx, hy = float(out[0]), float(out[1])

        # Escolha da ação
        if abs(hx) >= abs(hy):
            acao = 1 if hx > 0 else 3   # right / left
        else:
            acao = 0 if hy > 0 else 2   # up / down

        self.last_action = acao
        return acao