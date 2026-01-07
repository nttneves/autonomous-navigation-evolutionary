from agents.agent import Agent
from agents.qlearning_agent import QLearningAgent


class QLearningRuntimeAgent(Agent):
    """
    Wrapper para QLearningAgent para usar no Simulator/Renderer.
    NÃO aprende, só executa a política.
    """

    def __init__(self, agent: QLearningAgent, discretizer, id="q_agent"):
        super().__init__(id=id, sensores=True)

        self.agent = agent
        self.discretizer = discretizer
        self.current_state = None

    # --------------------------------------------------
    # Recebe observação do ambiente
    # --------------------------------------------------
    def observacao(self, obs):
        self.current_state = self.discretizer.discretize(obs)

    # --------------------------------------------------
    # Decide ação (CHAMADO PELO SIMULATOR)
    # --------------------------------------------------
    def age(self):
        if self.current_state is None:
            return 0  # ação default (ex: cima)

        state_idx = self.discretizer.tuple_to_index(self.current_state)
        return self.agent.choose_action(state_idx)

    # --------------------------------------------------
    # Compatibilidade com simulator
    # --------------------------------------------------
    def avaliacaoEstadoAtual(self, reward):
        pass

    def reset(self):
        self.current_state = None


