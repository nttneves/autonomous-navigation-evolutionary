# agents/evolved_agent.py
from agents.agent import Agent
import numpy as np
from model.model import create_mlp
import json

class EvolvedAgent(Agent):
    def __init__(self, id: str, model=None, dim_input_rn: int = 12, sensores: bool=True):
        super().__init__(id, politica="evolved", sensores=sensores)
        self.dim_input_rn = dim_input_rn

        # cria rede NumPy se não passada (agora 4 outputs)
        self.rede_neuronal = model if model is not None else create_mlp(
            input_dim=dim_input_rn,
            hidden_units=32,
            outputs=4
        )

        self.genoma = None

    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        agent_id = data["id"]
        dim_input = data.get("dim_input", 12)
        sensores = bool(data["sensores"])

        # cria novo modelo numpy com 4 outputs
        model = create_mlp(input_dim=dim_input, hidden_units=32, outputs=4)
        return cls(
            id=agent_id,
            model=model,
            dim_input_rn=dim_input,
            sensores=sensores
        )

    # ============================================================
    # GENOMA
    # ============================================================
    def set_genoma(self, genoma):
        self.genoma = genoma
        if genoma is not None:
            self.rede_neuronal.set_weights(genoma)

    # ============================================================
    # OBSERVAÇÃO -> VETOR
    # ============================================================
    def vetorizar_obs(self, obs):
        """
        O ambiente (MazeEnv) já devolve a observação como um vetor 1D de 12 inputs.
        Portanto, apenas devolvemos o vetor diretamente.
        """
        if obs is None:
            return np.zeros(self.dim_input_rn, dtype=np.float32)
        
        # O 'obs' JÁ É o array 1D com 12 valores.
        # NENHUMA CONVERSÃO É NECESSÁRIA.
        return obs

    # ============================================================
    # POLÍTICA DO AGENTE
    # ============================================================
    def age(self) -> int:
        vetor = self.vetorizar_obs(self.last_observation)
        out = self.rede_neuronal.forward(vetor.reshape(1, -1))[0]
        acao = int(np.argmax(out))  # escolhe ação com maior valor
        self.last_action = acao
        return acao

    # ============================================================
    # RESET PARA NOVO EPISÓDIO
    # ============================================================
    def reset(self):
        """Reinicia estado interno da RNN entre episódios."""
        if hasattr(self.rede_neuronal, 'reset_state'):
            self.rede_neuronal.reset_state()