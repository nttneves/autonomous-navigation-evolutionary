# model/model.py
import tensorflow as tf

tf.random.set_seed(42)

def create_rnn(input_dim=10, hidden_units=2, outputs=2):
    """
    RNN simples: input_dim -> Dense(internal) -> SimpleRNN(hidden_units) -> Dense(outputs)
    Usamos activations lineares nos outputs (interpretamos sinais).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        # mapeamento para dimensão interna antes do recurrent
        tf.keras.layers.Dense(hidden_units, activation='tanh'),
        # SimpleRNN em modo stateful=False (estado gerido pelo input único)
        tf.keras.layers.Reshape((1, hidden_units)),  # faz shape (batch, timesteps=1, features)
        tf.keras.layers.SimpleRNN(hidden_units, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(outputs, activation='linear')
    ])
    return model

# compatibilidade: se o resto do código esperar create_mlp
def create_mlp(input_dim=10):
    return create_rnn(input_dim=input_dim)