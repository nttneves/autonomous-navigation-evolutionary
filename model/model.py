import tensorflow as tf

tf.random.set_seed(42)

def create_mlp(input_dim=10):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # <-- RNN precisa de formato 3D
        tf.keras.layers.SimpleRNN(
            units=2,
            activation='tanh',
            return_sequences=False
        ),
        tf.keras.layers.Dense(4, activation='softmax')
    ])