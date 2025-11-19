# environment_farol.py
"""
FarolEnv: ambiente "Lighthouse" compatível com a API do teu projeto.

Fornece:
- FarolEnv(Enviroment): ambiente 2D simples com um farol (marcado como RECURSO)
- make_farol_env_runner(farol_env, start_pos, max_steps): retorna função env_runner(model)
    -> env_runner(model) devolve (bc, traj, rewards)
- plot_rewards(rewards, filename): plota e guarda gráfico do reward por passo / acumulado
"""

import numpy as np
from typing import Tuple
from copy import deepcopy
import matplotlib.pyplot as plt

# caminho para enunciado (ficheiro que carregaste) — se quiseres abrir/usar
ENUNCIADO_PATH = "/mnt/data/Enunciado Projeto_ Simulador de SMA (1).pdf"

# IMPORTA a base do teu ambiente (ajusta se o ficheiro estiver noutro local)
try:
    from environment import Enviroment, PAREDE, VAZIO, RECURSO, NINHO, AGENTE
except Exception:
    # fallback: define constantes mínimas se a importação falhar
    from environment import Enviroment  # deixar levantar erro se nem existir
    PAREDE = -1
    VAZIO = 0
    RECURSO = 1
    NINHO = 2
    AGENTE = 3

# Ações mapeadas para deslocamento (x, y)
ACTION_TO_DELTA = {
    0: (0, -1),   # UP / N
    1: (1, 0),    # RIGHT / E
    2: (0, 1),    # DOWN / S
    3: (-1, 0),   # LEFT / W
}


class FarolEnv(Enviroment):
    def __init__(self,
                 mapa_file: str = None,
                 tamanho: Tuple[int, int] = (20, 20),
                 farol_pos: Tuple[int, int] = None,
                 max_steps: int = 200):
        """
        mapa_file: opcional; se None cria mapa vazio
        tamanho: (width, height)
        farol_pos: (x,y) do farol; se None, centro do mapa
        max_steps: limite de passos por episódio
        """
        self.max_steps = int(max_steps)
        self.farol_pos = farol_pos
        super().__init__(mapa_file, tamanho)
        self._initial_map = self.mapa_estado.copy()
        self._steps = 0
        self._done = False

    def _carregar_mapa(self, mapa_file: str) -> np.ndarray:
        w, h = self.tamanho
        mapa = np.full((h, w), VAZIO, dtype=int)

        if mapa_file:
            try:
                with open(mapa_file, 'r') as f:
                    lines = [l.rstrip('\n') for l in f if l.strip()]
                for y, line in enumerate(lines):
                    for x, ch in enumerate(line):
                        if x >= w or y >= h:
                            continue
                        if ch == '#':
                            mapa[y, x] = PAREDE
                        else:
                            mapa[y, x] = VAZIO
            except Exception:
                mapa = np.full((h, w), VAZIO, dtype=int)

        if self.farol_pos is None:
            self.farol_pos = (w // 2, h // 2)
        fx, fy = self.farol_pos
        # marca farol como RECURSO para os sensores detectarem
        mapa[fy, fx] = RECURSO
        return mapa

    # Override regista_agente para garantir agente.posicao existe e está sincronizado
    def regista_agente(self, agente, pos_inicial: tuple):
        super().regista_agente(agente, pos_inicial)
        # garante que o objeto agente tem atributo posicao compatível com sensores
        try:
            agente.posicao = tuple(pos_inicial)
        except Exception:
            pass

    def reset(self):
        """Reinicia mapa e contadores; mantém farol_pos."""
        self.mapa_estado = self._initial_map.copy()
        fx, fy = self.farol_pos
        self.mapa_estado[fy, fx] = RECURSO
        self.posicoes_agentes = {}
        self.passo_tempo = 0
        self._steps = 0
        self._done = False

    def atualizacao(self):
        """Avança o relógio do ambiente."""
        self.passo_tempo += 1
        self._steps += 1

    def agir(self, accao: int, agente):
        """
        Aplica ação do agente e devolve (reward: float, done: bool, info: dict)
        """
        if agente.id not in self.posicoes_agentes:
            return 0.0, False, {}

        x, y = self.posicoes_agentes[agente.id]
        dx, dy = ACTION_TO_DELTA[int(accao)]
        nx, ny = x + dx, y + dy

        info = {}
        reward = 0.0
        done = False

        # colisão ou fora
        if not (0 <= nx < self.tamanho[0] and 0 <= ny < self.tamanho[1]) or self.mapa_estado[ny, nx] == PAREDE:
            reward = -0.01
            info['collision'] = True
            nx, ny = x, y
        else:
            # move o agente e sincroniza agente.posicao
            self.posicoes_agentes[agente.id] = (nx, ny)
            try:
                agente.posicao = (nx, ny)
            except Exception:
                pass

        # verificar se atingiu o farol
        fx, fy = self.farol_pos
        if (nx, ny) == (fx, fy):
            reward += 1.0
            done = True
            info['reached_beacon'] = True

        # atualiza agente (se implementa contabilização de reward)
        try:
            agente.avaliacaoEstadoAtual(reward)
        except Exception:
            pass

        return float(reward), bool(done), info

    def terminou(self):
        """Condição de fim."""
        if self._done:
            return True
        if self._steps >= self.max_steps:
            return True
        for pos in self.posicoes_agentes.values():
            if pos == self.farol_pos:
                return True
        return False

    def tem_obstaculo_ou_objeto_relevante(self, x: int, y: int) -> bool:
        if not (0 <= x < self.tamanho[0] and 0 <= y < self.tamanho[1]):
            return True
        cel = self.mapa_estado[y, x]
        if cel == PAREDE:
            return True
        if cel == RECURSO:
            return True
        return False


# ---------------------------------------------------------------------
# env_runner + plotting helpers
# ---------------------------------------------------------------------
def make_farol_env_runner(farol_env: FarolEnv, start_pos=(1, 1), max_steps=None):
    """
    Retorna uma função env_runner(model) -> (bc, traj, rewards)
    - bc: np.array([x_norm, y_norm, reached_flag, steps_norm])
    - traj: lista de posições (x,y) ao longo do episódio
    - rewards: lista de rewards por passo
    """
    max_steps = int(max_steps) if max_steps is not None else getattr(farol_env, "max_steps", 200)

    def env_runner(model):
        env = deepcopy(farol_env)
        if hasattr(env, "reset"):
            env.reset()

        # tentativa de importar classes do teu projecto (Agente e sensores)
        try:
            # ajusta caminhos se necessário
            from agent import Agent as AgentClass
        except Exception:
            AgentClass = None

        try:
            from sensor.sensor_objeto import sensor_objeto as SensorObjetoClass
            from sensor.sensor_obstaculo import sensor_obstaculo as SensorObstaculoClass
        except Exception:
            SensorObjetoClass = None
            SensorObstaculoClass = None

        # Cria uma classe Agente mínima se não existir Agent no projeto
        if AgentClass is None:
            class AgentClass:
                def __init__(self, id, politica_model=None, sensor_order=None):
                    self.id = id
                    self.politica = politica_model
                    self.sensores = {}
                    self.sensor_order = sensor_order or []
                    self.posicao = None
                    self.rewards = []

                def instala(self, nome, sensor):
                    self.sensores[nome] = sensor
                    if nome not in self.sensor_order:
                        self.sensor_order.append(nome)

                def observação(self, obs):
                    self.ultima_observacao = obs

                def _obs_to_input(self):
                    parts = []
                    for nome in self.sensor_order:
                        arr = self.ultima_observacao.get(nome)
                        if arr is None:
                            dim = getattr(self.sensores.get(nome), 'dim', None) or (8 if 'objeto' in nome else 4)
                            arr = np.zeros(dim, dtype=np.float32)
                        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                        sensor = self.sensores.get(nome)
                        if sensor is not None and hasattr(sensor, 'alcance') and sensor.alcance:
                            arr = arr.astype(np.float32) / float(sensor.alcance)
                        parts.append(arr)
                    if not parts:
                        return np.zeros((1, 0), dtype=np.float32)
                    return np.concatenate(parts, axis=0).reshape(1, -1)

                def age(self, stochastic=False):
                    # assume política com API model(x).numpy()
                    x = self._obs_to_input()
                    probs = self.politica(x).numpy().squeeze()
                    if stochastic:
                        return int(np.random.choice(len(probs), p=probs))
                    return int(np.argmax(probs))

                def avaliacaoEstadoAtual(self, recompensa):
                    try:
                        self.rewards.append(float(recompensa))
                    except Exception:
                        pass

        # Instancia sensores fallback se necessário
        if SensorObjetoClass is None:
            class SensorObjetoClass:
                def __init__(self, alcance=10):
                    self.alcance = alcance
                    self.dim = 8

                def gerar_observacao(self, ambiente, agente):
                    # simples: devolve alcance em todas as direções (sem deteção)
                    return np.full(8, self.alcance, dtype=int)

        if SensorObstaculoClass is None:
            class SensorObstaculoClass:
                def __init__(self, alcance=10):
                    self.alcance = alcance
                    self.dim = 4

                def gerar_observacao(self, ambiente, agente):
                    return np.full(4, self.alcance, dtype=int)

        # Cria agente com política (model) e sensores
        agent = AgentClass(id='ag1', politica_model=model, sensor_order=['sensor_objeto', 'sensor_obstaculo']) \
            if AgentClass is not None else AgentClass('ag1', politica_model=model, sensor_order=['sensor_objeto', 'sensor_obstaculo'])

        # Se AgentClass veio do projeto, não se esqueça de adaptar os nomes de construtor
        try:
            # se a classe do projeto usa outro nome de param, tenta ajustar
            if hasattr(agent, "instala"):
                s_obj = SensorObjetoClass(alcance=10); s_obj.dim = 8
                s_obs = SensorObstaculoClass(alcance=10); s_obs.dim = 4
                agent.instala('sensor_objeto', s_obj)
                agent.instala('sensor_obstaculo', s_obs)
        except Exception:
            # ignore se não for compatível
            pass

        # regista agente e define posição inicial
        env.regista_agente(agent, pos_inicial=start_pos)
        env.posicoes_agentes[agent.id] = start_pos
        try:
            agent.posicao = tuple(start_pos)
        except Exception:
            pass

        traj = [start_pos]
        rewards = []
        steps = 0
        reached = 0

        while steps < max_steps and not env.terminou():
            obs = env.observacaoPara(agent)
            # garante que agente usa o método correcto para receber observação
            try:
                if hasattr(agent, "observação"):
                    agent.observação(obs)
                elif hasattr(agent, "observacao"):
                    agent.observacao(obs)
                else:
                    agent.ultima_observacao = obs
            except Exception:
                agent.ultima_observacao = obs

            # escolher ação
            try:
                accao_idx = agent.age(stochastic=False)
            except Exception:
                # fallback: escolha aleatória entre 4 ações
                accao_idx = int(np.random.randint(0, 4))

            # aplicar ação
            out = env.agir(accao_idx, agent)
            # aceitar diferentes assinaturas: (reward, done, info) ou (reward, done)
            if isinstance(out, tuple):
                if len(out) == 3:
                    reward, done, info = out
                elif len(out) == 2:
                    reward, done = out
                    info = {}
                else:
                    reward, done, info = 0.0, False, {}
            else:
                reward, done, info = 0.0, False, {}

            # regista reward
            rewards.append(float(reward))

            # atualiza ambiente
            if hasattr(env, "atualizacao"):
                env.atualizacao()

            steps += 1
            traj.append(env.get_posicao_agente(agent))

            if info.get('reached_beacon', False):
                reached = 1
                break

        final_pos = env.get_posicao_agente(agent)
        w, h = env.tamanho
        bc = np.array([final_pos[0] / float(w), final_pos[1] / float(h), float(reached), steps / float(max_steps)], dtype=np.float32)

        return bc, traj, rewards

    return env_runner


def plot_rewards(rewards, filename="reward_plot.png", save=True, show=True):
    """
    Plota:
     - reward por passo (barra/linha)
     - reward acumulado ao longo do tempo
    Guarda imagem em filename se save=True.
    """
    if rewards is None or len(rewards) == 0:
        raise ValueError("Lista de rewards vazia")

    steps = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards, dtype=np.float32)
    cum_rewards = np.cumsum(rewards)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, rewards, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward por passo")

    plt.subplot(1, 2, 2)
    plt.plot(steps, cum_rewards, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Reward acumulado")
    plt.title("Reward acumulado ao longo do episódio")

    plt.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close()
    return filename
