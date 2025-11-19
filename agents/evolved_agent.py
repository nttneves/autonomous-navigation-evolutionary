import numpy as np
from agents.agent import Agent
from model import create_mlp
from algorithms.genetic import set_weights_vector

class EvolvedAgent(Agent):
    """
    Agente cuja política é dada por uma rede neuronal evoluída.
    Recebe a observação diretamente no método age(obs).
    """

    def __init__(self, id: str, model=None, dim_input_rn: int = None):
        super().__init__(id)

        if model is not None:
            self.rede_neuronal = model
        else:
            if dim_input_rn is None:
                raise ValueError("Specify either model or dim_input_rn")
            self.rede_neuronal = create_mlp(dim_input_rn)

        self.genoma = None
        self.last_action = None

    @classmethod
    def cria(cls, ficheiro_json: str):
        import json
        with open(ficheiro_json, "r") as f:
            data = json.load(f)

        return cls(
            id=data["id"],
            dim_input_rn=data["dim_input"]
        )
    
    def observacao(self, obs):
        """Compatível com o método abstrato da classe base."""
        self.last_observation = obs

    def set_genoma(self, genoma):
        """Atualizar pesos da rede a partir de vetor plano."""
        self.genoma = genoma
        set_weights_vector(self.rede_neuronal, genoma)

    def _vetorizar_obs(self, obs):
        """Converte observação em vetor 1D."""
        if obs is None:
            return np.zeros(1)

        if isinstance(obs, dict):
            partes = []
            for v in obs.values():
                partes.append(np.array(v).flatten())
            return np.concatenate(partes)

        return np.array(obs).flatten()

    def age(self, obs):
        """
        Calcula a ação baseada na rede neuronal.
        O simulador chama: a = agent.age(obs)
        """
        vetor = self._vetorizar_obs(obs)

        out = self.rede_neuronal(
            np.array([vetor], dtype=np.float32),
            training=False
        )

        acao = int(np.argmax(out.numpy()))
        self.last_action = acao
        return acao

    def avaliacaoEstadoAtual(self, recompensa: float):
        """Regista reward (interface exigida pelo enunciado)."""
        self.regista_reward(recompensa)