# agents/qlearning_agent.py
import numpy as np
import random
import pickle
from typing import Optional, Tuple, Iterable
from agents.agent import Agent


class QLearningAgent(Agent):

    def __init__(
        self,
        id: str,
        input_dim: int = 12,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        n_bins: int = 10,
        sensores: bool = True,
        optimistic_init: float = 0.0,
        use_hash_keys: bool = False,
        min_explore_steps: int = 0,
        debug: bool = False
    ):
        super().__init__(id, politica="qlearning", sensores=sensores)

        self.input_dim = int(input_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.n_bins = int(n_bins)
        self.n_actions = 4
        self.optimistic_init = float(optimistic_init)
        self.use_hash_keys = bool(use_hash_keys)
        self.min_explore_steps = int(min_explore_steps)
        self.debug = bool(debug)

        self.q_table = {}

        self.last_observation = None
        self.last_action = None
        self.total_steps = 0

        self.obs_min = np.full(self.input_dim, np.inf, dtype=np.float32)
        self.obs_max = np.full(self.input_dim, -np.inf, dtype=np.float32)
        self.eps = 1e-8

        self.bins = None
        self._bins_ready = False

    def _update_obs_minmax(self, obs: np.ndarray):
        if obs is None:
            return
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[0] != self.input_dim:
            return
        self.obs_min = np.minimum(self.obs_min, obs)
        self.obs_max = np.maximum(self.obs_max, obs)
        if np.all(self.obs_max - self.obs_min > 1e-6):
            self._build_bins()

    def _build_bins(self):
        self.bins = []
        for i in range(self.input_dim):
            low = float(self.obs_min[i])
            high = float(self.obs_max[i])
            if high - low < 1e-6:
                self.bins.append(np.array([]))
            else:
                edges = np.linspace(low, high, self.n_bins + 1)[1:-1]
                self.bins.append(edges)
        self._bins_ready = True
        if self.debug:
            print("[QLAgent] Bins construídos.")

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        denom = (self.obs_max - self.obs_min) + self.eps
        norm = (obs - self.obs_min) / denom
        return np.clip(norm, 0.0, 1.0)

    def _discretize_state(self, observation_vector: Iterable) -> Optional[Tuple]:
        if observation_vector is None:
            return None

        vec = np.asarray(observation_vector, dtype=np.float32)
        if vec.size != self.input_dim:
            return None

        self._update_obs_minmax(vec)

        if not self._bins_ready:
            denom = (np.ptp(vec) + self.eps)
            if denom == 0:
                idxs = tuple(0 for _ in range(self.input_dim))
            else:
                norm = np.clip((vec - vec.min()) / denom, 0.0, 1.0)
                idxs = tuple(int(np.floor(v * (self.n_bins - 1))) for v in norm)
        else:
            idxs = []
            for i, v in enumerate(vec):
                if self.bins[i].size == 0:
                    idxs.append(0)
                else:
                    bin_idx = int(np.digitize(float(v), self.bins[i]))
                    idxs.append(min(bin_idx, self.n_bins - 1))
            idxs = tuple(idxs)

        if self.use_hash_keys:
            return hash(idxs)
        return tuple(idxs)

    def vetorizar_obs(self, obs):
        if obs is None:
            return None
        arr = np.asarray(obs, dtype=np.float32)
        if arr.size != self.input_dim:
            try:
                arr = arr.flatten()
                if arr.size < self.input_dim:
                    tmp = np.zeros(self.input_dim, dtype=np.float32)
                    tmp[:arr.size] = arr
                    arr = tmp
                else:
                    arr = arr[:self.input_dim]
            except Exception:
                return None
        return arr

    def observacao(self, obs):
        vec = self.vetorizar_obs(obs)
        if vec is not None:
            self._update_obs_minmax(vec)
            self.last_observation = vec

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.full(self.n_actions, self.optimistic_init, dtype=np.float32)
        return self.q_table[state_key]

    def age(self) -> int:
        self.total_steps += 1

        if self.last_observation is None:
            return random.randint(0, self.n_actions - 1)

        state_key = self._discretize_state(self.last_observation)
        if state_key is None:
            return random.randint(0, self.n_actions - 1)

        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            q_values = self.get_q_values(state_key)
            action = int(np.argmax(q_values))

        self.last_action = action
        if self.debug:
            print(f"[QLAgent] age -> state={state_key}, action={action}, qsize={len(self.q_table)}")
        return action

    def avaliacaoEstadoAtual(self, reward: float):
        pass

    def update_q_value(self, state, action, reward, next_state, done):
        state_key = self._discretize_state(state)
        if state_key is None:
            return

        if action is None or not (0 <= action < self.n_actions):
            return

        current_q = float(self.get_q_values(state_key)[action])

        if done or next_state is None:
            target_q = reward
        else:
            next_key = self._discretize_state(next_state)
            if next_key is None:
                target_q = reward
            else:
                next_qs = self.get_q_values(next_key)
                max_next_q = float(np.max(next_qs))
                target_q = reward + self.gamma * max_next_q

        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state_key][action] = new_q

        if self.debug:
            print(f"[QLAgent] update s={state_key} a={action} r={reward:.3f} curQ={current_q:.4f} newQ={new_q:.4f}")

    def decay_epsilon(self):
        if self.total_steps < self.min_explore_steps:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, float(self.epsilon * self.epsilon_decay))

    def reset(self):
        self.last_observation = None
        self.last_action = None

    def save(self, path: str):
        data = {
            'q_table': self.q_table,
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'n_bins': self.n_bins,
            'optimistic_init': self.optimistic_init,
            'use_hash_keys': self.use_hash_keys,
            'obs_min': self.obs_min.tolist(),
            'obs_max': self.obs_max.tolist(),
            'bins': [b.tolist() for b in (self.bins or [])] if self.bins is not None else None
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, id: str = "loaded"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = cls(
            id=id,
            input_dim=data.get('input_dim', 12),
            learning_rate=data.get('learning_rate', 0.1),
            gamma=data.get('gamma', 0.99),
            epsilon=data.get('epsilon', 0.0),
            epsilon_min=data.get('epsilon_min', 0.01),
            epsilon_decay=data.get('epsilon_decay', 0.995),
            n_bins=data.get('n_bins', 10),
            optimistic_init=data.get('optimistic_init', 0.0),
            use_hash_keys=data.get('use_hash_keys', False),
        )
        agent.q_table = data.get('q_table', {})
        agent.obs_min = np.array(data.get('obs_min', agent.obs_min), dtype=np.float32)
        agent.obs_max = np.array(data.get('obs_max', agent.obs_max), dtype=np.float32)
        bins = data.get('bins', None)
        if bins is not None:
            agent.bins = [np.array(b, dtype=np.float32) for b in bins]
            agent._bins_ready = True
        return agent

    def state_size_estimate(self):
        try:
            return (self.n_bins ** self.input_dim)
        except OverflowError:
            return float('inf')