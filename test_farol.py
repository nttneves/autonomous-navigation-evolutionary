# test_farol.py
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

try:
    from environment_farol import FarolEnv, make_farol_env_runner
except Exception as e:
    raise RuntimeError("Não consegui importar FarolEnv. Verifica environment_farol.py") from e

# tenta importar sensores e Agente do teu projeto; se falhar, usa implementações fallback simples
try:
    from sensor.sensor_objeto import sensor_objeto as SensorObjetoClass
    from sensor.sensor_obstaculo import sensor_obstaculo as SensorObstaculoClass
except Exception:
    # fallback: sensores simples com a mesma interface (gerar_observacao)
    class SensorObjetoClass:
        def __init__(self, alcance=10):
            self.alcance = alcance
            self.dim = 8
        def gerar_observacao(self, ambiente, agente):
            # devolve distância máxima (sem deteção) para simplificar fallback
            return np.full(8, self.alcance, dtype=int)

    class SensorObstaculoClass:
        def __init__(self, alcance=10):
            self.alcance = alcance
            self.dim = 4
        def gerar_observacao(self, ambiente, agente):
            return np.full(4, self.alcance, dtype=int)

# tenta importar Agente; se não existir, define a versão robusta aqui
try:
    from agent import Agente
except Exception:
    import numpy as _np
    class Agente:
        def __init__(self, id, politica_model=None, sensor_order=None):
            self.id = id
            self.politica = politica_model
            self.sensores = {}
            self.sensor_order = sensor_order or []
            self.recompensa_acumulada = 0.0
            self.ultima_accao = None
            self.ultima_observacao = None

        def instala(self, nome, sensor):
            self.sensores[nome] = sensor
            if nome not in self.sensor_order:
                self.sensor_order.append(nome)

        def set_sensor_order(self, ordem):
            self.sensor_order = list(ordem)

        def observação(self, obs: dict):
            self.ultima_observacao = obs

        def _obs_to_input(self):
            parts = []
            for nome in self.sensor_order:
                arr = self.ultima_observacao.get(nome)
                if arr is None:
                    # preencher com zeros com base numa hipótese de dimensão
                    dim = getattr(self.sensores.get(nome), 'dim', None) or (8 if 'objeto' in nome else 4)
                    arr = _np.zeros(dim, dtype=_np.float32)
                arr = _np.asarray(arr, dtype=_np.float32).reshape(-1)
                # normalizar se sensor tiver alcance
                sensor = self.sensores.get(nome)
                if sensor is not None and hasattr(sensor, 'alcance') and sensor.alcance:
                    arr = arr.astype(_np.float32) / float(sensor.alcance)
                parts.append(arr)
            if not parts:
                return _np.zeros((1,0), dtype=_np.float32)
            return _np.concatenate(parts, axis=0).reshape(1, -1)

        def age(self, stochastic=False):
            if self.politica is None:
                raise RuntimeError("Agente sem política definida")
            x = self._obs_to_input()
            probs = self.politica(x).numpy().squeeze()
            if stochastic:
                accao_idx = int(np.random.choice(len(probs), p=probs))
            else:
                accao_idx = int(np.argmax(probs))
            self.ultima_accao = accao_idx
            return accao_idx

        def avaliacaoEstadoAtual(self, recompensa):
            self.recompensa_acumulada += float(recompensa)

        def comunica(self, mensagem, de_agente):
            pass

# função create_mlp (igual à tua)
import tensorflow as tf
tf.random.set_seed(42)
def create_mlp(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model

# --- Test runner simples para o Farol ---
def test_single_episode():
    # cria ambiente
    tamanho = (20, 15)
    farol_pos = (15, 7)
    env = FarolEnv(mapa_file=None, tamanho=tamanho, farol_pos=farol_pos, max_steps=200)
    env.reset()

    # cria modelo com input_dim = 8 + 4 = 12
    input_dim = 12
    model = create_mlp(input_dim)

    # cria agente e instala sensores
    agent = Agente(id='ag1', politica_model=model, sensor_order=['sensor_objeto', 'sensor_obstaculo'])
    s_obj = SensorObjetoClass(alcance=10)
    s_obj.dim = 8
    s_obs = SensorObstaculoClass(alcance=10)
    s_obs.dim = 4
    agent.instala('sensor_objeto', s_obj)
    agent.instala('sensor_obstaculo', s_obs)

    # regista agente e define posição inicial
    start = (1, 1)
    env.regista_agente(agent, pos_inicial=start)
    env.posicoes_agentes[agent.id] = start

    traj = [start]
    total_reward = 0.0
    steps = 0
    reached = False

    while steps < env.max_steps and not env.terminou():
        obs = env.observacaoPara(agent)
        agent.observação(obs)
        accao_idx = agent.age(stochastic=False)
        reward, done, info = env.agir(accao_idx, agent)
        env.atualizacao()
        total_reward += reward
        steps += 1
        traj.append(env.get_posicao_agente(agent))
        if info.get('reached_beacon', False):
            reached = True
            break

    final_pos = env.get_posicao_agente(agent)
    print("RESULTS:")
    print("  reached:", reached)
    print("  steps:", steps)
    print("  total_reward:", total_reward)
    print("  final_pos:", final_pos)
    # BC: [x/w, y/h, reached, steps/max_steps]
    w, h = env.tamanho
    bc = np.array([final_pos[0]/w, final_pos[1]/h, float(reached), steps / env.max_steps], dtype=np.float32)
    print("  BC:", bc)

    # plota trajectoria
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    plt.figure(figsize=(6,5))
    plt.title("Trajetória do agente no Farol")
    # imagem do mapa (trocar valores para visual)
    mapa_vis = env.mapa_estado.copy()
    plt.imshow(mapa_vis, origin='upper')
    plt.plot(xs, ys, marker='o', linewidth=2)
    plt.scatter([farol_pos[0]], [farol_pos[1]], marker='*', s=150, c='yellow', label='Farol')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

    return bc, traj

if __name__ == "__main__":
    bc, traj = test_single_episode()
    # grava bc para inspeção
    np.save("last_bc.npy", bc)
    print("BC guardado em last_bc.npy")
