# agents/qlearning_agent.py
import numpy as np
import random
import pickle

class QLearningAgent:
    """
    Agente de aprendizagem (Q-table).
    NÃO interage diretamente com o ambiente.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    # --------------------------------------------------
    # Política
    # --------------------------------------------------

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploração
            return np.random.randint(self.n_actions)
        else:
            # Exploração: pegar ações de valor máximo
            q_values = self.Q[state]
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)

    # --------------------------------------------------
    # Aprendizagem
    # --------------------------------------------------

    def update(self, s, a, r, s2, done):
        best_next = np.max(self.Q[s2])
        target = r + self.gamma * best_next * (not done)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --------------------------------------------------
    # Persistência
    # --------------------------------------------------

    def save(self, path: str):
        data = {
            'Q': self.Q.tolist(),
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        n_states = len(data['Q'])

        agent = cls(
            n_states=n_states,
            n_actions=data.get('n_actions', 4),
            alpha=data.get('alpha', 0.1),
            gamma=data.get('gamma', 0.99),
            epsilon=data.get('epsilon', 0.0),
            epsilon_min=data.get('epsilon_min', 0.05),
            epsilon_decay=data.get('epsilon_decay', 0.995)
        )

        agent.Q = np.array(data['Q'], dtype=np.float32)
        return agent
    

    def observacao(self, obs):
        # guarda estado discreto
        self.current_state = self.discretizer.discretize(obs)

    def avaliacaoEstadoAtual(self, reward):
        pass  # opcional, só para compatibilidade

    def reset(self):
        self.epsilon = 0.0
