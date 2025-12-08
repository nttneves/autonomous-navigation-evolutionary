# model/model.py
import numpy as np

class SimpleRNNCell:
    """Célula RNN simples: h = tanh(x @ Wx + h_prev @ Wh + b)"""
    def __init__(self, input_dim, hidden_units, seed=42):
        rng = np.random.default_rng(seed)
        self.Wx = rng.normal(0, 0.1, size=(input_dim, hidden_units))
        self.Wh = rng.normal(0, 0.1, size=(hidden_units, hidden_units))
        self.b  = np.zeros(hidden_units)
        self.h_prev = None  # estado anterior da RNN

    def forward(self, x):
        batch = x.shape[0]
        if self.h_prev is None:
            self.h_prev = np.zeros((batch, self.Wh.shape[0]))
        h = np.tanh(x @ self.Wx + self.h_prev @ self.Wh + self.b)
        self.h_prev = h
        return h

    def reset_state(self):
        self.h_prev = None


class SimpleMLP_RNN:
    """Input -> Dense -> RNN -> Dense -> Output"""
    def __init__(self, input_dim=12, hidden_units=32, outputs=4, seed=42):
        rng = np.random.default_rng(seed)

        # Dense inicial
        self.W1 = rng.normal(0, 0.1, size=(input_dim, hidden_units))
        self.b1 = np.zeros(hidden_units)

        # RNN
        self.rnn = SimpleRNNCell(hidden_units, hidden_units, seed=seed + 1)

        # Dense final (linear)
        self.W2 = rng.normal(0, 0.1, size=(hidden_units, outputs))
        self.b2 = np.zeros(outputs)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        h = self.rnn.forward(h)
        y = h @ self.W2 + self.b2
        return y

    def reset_state(self):
        """Reset do estado interno da RNN"""
        self.rnn.reset_state()

    # ===========================================================
    # Pesos para o GA
    # ===========================================================
    def get_weights(self):
        return [self.W1, self.b1, self.rnn.Wx, self.rnn.Wh, self.rnn.b, self.W2, self.b2]

    def set_weights(self, weights):
        self.W1, self.b1, self.rnn.Wx, self.rnn.Wh, self.rnn.b, self.W2, self.b2 = weights


def create_mlp(input_dim=12, hidden_units=32, outputs=4):
    return SimpleMLP_RNN(input_dim, hidden_units, outputs)