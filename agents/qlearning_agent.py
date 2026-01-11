import numpy as np
import random
import pickle
from agents.agent import Agent

class QLearningAgent(Agent):
    """
    Q-Learning completo:
    - aprende
    - executa
    - guarda/carrega Q-table
    """

    def __init__(
        self,
        id: str,
        discretizer,
        n_states: int,
        n_actions: int,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        super().__init__(id=id, sensores=True)

        self.discretizer = discretizer
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

        self.state = None
        self.last_state = None
        self.last_action = None

    # ==================================================
    # Observação do ambiente
    # ==================================================
    def observacao(self, obs):
        disc = self.discretizer.discretize(obs)
        self.state = self.discretizer.tuple_to_index(disc)

    # ==================================================
    # Escolha de ação (chamado pelo Simulator)
    # ==================================================
    def age(self):
        if self.state is None:
            return 0

        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            qvals = self.Q[self.state]
            max_q = np.max(qvals)
            action = random.choice(np.where(qvals == max_q)[0])

        self.last_state = self.state
        self.last_action = action
        return action

    # ==================================================
    # Aprendizagem
    # ==================================================
    def avaliacaoEstadoAtual(self, reward, done=False):
        if self.last_state is None or self.state is None:
            return

        best_next = np.max(self.Q[self.state])
        target = reward + self.gamma * best_next * (not done)
        self.Q[self.last_state, self.last_action] += \
            self.alpha * (target - self.Q[self.last_state, self.last_action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.state = None
        self.last_state = None
        self.last_action = None

    # ==================================================
    # Persistência
    # ==================================================
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "Q": self.Q,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "n_actions": self.n_actions
            }, f)

    @classmethod
    def load(cls, path: str, discretizer, id="q_loaded"):
        with open(path, "rb") as f:
            data = pickle.load(f)

        Q = np.array(data["Q"], dtype=np.float32)
        n_states, n_actions = Q.shape

        agent = cls(
            id=id,
            discretizer=discretizer,
            n_states=n_states,
            n_actions=n_actions,
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.99),
            epsilon=0.0,              # 🔒 avaliação pura
            epsilon_min=0.0,
            epsilon_decay=1.0
        )

        agent.Q = Q
        return agent