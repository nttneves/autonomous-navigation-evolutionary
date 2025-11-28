# environments/environment_maze.py
from typing import Tuple
from environments.environment import Enviroment
import numpy as np
import random
from agents.agent import Agent
import math

ACTION_TO_DELTA = {
    0: (0, -1),   # up
    1: (1, 0),    # right
    2: (0, 1),    # down
    3: (-1, 0),   # left
}

VAZIO = 0
PAREDE = 1
SAIDA = 2


class MazeEnv(Enviroment):

    def __init__(self,
                 dificuldade: int = 0,
                 max_steps: int = 200,
                 range_max: int = 5,
                 seed: int | None = None):

        if dificuldade == 0:
            tamanho = (21, 21)
            self.seed = 42
        elif dificuldade == 1:
            tamanho = (31, 31)
            self.seed = 150
        else:
            tamanho = (41, 41)
            self.seed = 456

        super().__init__(tamanho=tamanho, dificuldade=dificuldade)

        self.max_steps = max_steps
        self.range_max = range_max
        self.saida_pos = None

        # gerar labirinto determinístico
        self.mapa_estado, self.saida_pos = self.criar_mapa()

        self._initial_map = self.mapa_estado.copy()
        self._steps = 0


    # ------------------------------------------------------------------
    def reset(self):
        self.mapa_estado = self._initial_map.copy()

        # marca a saída
        bx, by = self.saida_pos
        self.mapa_estado[by, bx] = SAIDA

        self.posicoes_agentes = {}
        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------

    def criar_mapa(self):
        w, h = self.tamanho

        rng = random.Random(self.seed)

        # garantir ímpares
        if w % 2 == 0: w -= 1
        if h % 2 == 0: h -= 1

        maze = np.full((h, w), fill_value=PAREDE, dtype=int)

        # algoritmo DFS
        def carve(x, y):
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            rng.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < w-1 and 1 <= ny < h-1 and maze[ny, nx] == PAREDE:
                    maze[ny, nx] = VAZIO
                    maze[y + dy//2, x + dx//2] = VAZIO
                    carve(nx, ny)

        # ponto inicial fixo
        maze[1, 1] = VAZIO
        carve(1, 1)

        # escolher saída entre células abertas da linha 1
        candidatos = [x for x in range(1, w-1) if maze[1, x] == VAZIO]

        if candidatos:
            sx = rng.choice(candidatos)
        else:
            sx = w // 2  # fallback

        saida_pos = (sx, 0)
        maze[0, sx] = SAIDA

        # spawn garantido
        maze[h-1, 1] = VAZIO
        maze[h-2, 1] = VAZIO
        maze[h-1, 2] = VAZIO

        return maze, saida_pos
        
    # ------------------------------------------------------------------
    def _in_bounds(self, x, y):
        w, h = self.tamanho
        return 0 <= x < w and 0 <= y < h


    # ------------------------------------------------------------------
    def _ray_distance(self, x0, y0, dx, dy, max_range=None):
        if max_range is None:
            max_range = self.range_max

        x, y = x0, y0
        for step in range(1, max_range + 1):
            nx = x + dx * step
            ny = y + dy * step
            if not self._in_bounds(nx, ny):
                return (step - 1) / max_range
            if self.mapa_estado[ny, nx] == PAREDE:
                return (step - 1) / max_range
        return 1.0


    # ------------------------------------------------------------------
    def _radar_quadrants(self, agente: Agent):
        ax, ay = agente.posicao
        bx, by = self.saida_pos

        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:
            return [0, 0, 0, 0]

        if abs(dy) >= abs(dx):
            return [1, 0, 0, 0] if dy < 0 else [0, 0, 1, 0]
        else:
            return [0, 1, 0, 0] if dx > 0 else [0, 0, 0, 1]


    # ------------------------------------------------------------------
    def observacaoPara(self, agente: Agent):
        ax, ay = agente.posicao
        max_r = self.range_max if agente.sensores else 1

        rf_dirs = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
            (1, -1),
            (-1, -1),
        ]

        ranges = [self._ray_distance(ax, ay, dx, dy, max_range=max_r) for dx, dy in rf_dirs]
        radar = self._radar_quadrants(agente)

        return {"ranges": ranges, "radar": radar}


    # ------------------------------------------------------------------
    def atualizacao(self):
        self._steps += 1


    # ------------------------------------------------------------------
    def agir(self, accao: int, agente: Agent):
        x, y = self.posicoes_agentes[agente.id]
        dx, dy = ACTION_TO_DELTA[accao]
        nx, ny = x + dx, y + dy

        reward = 0.0
        done = False
        info = {}

        if not self._in_bounds(nx, ny) or self.mapa_estado[ny, nx] == PAREDE:
            reward -= 0.01
            nx, ny = x, y
        else:
            self.posicoes_agentes[agente.id] = (nx, ny)
            agente.posicao = (nx, ny)

        if (nx, ny) == self.saida_pos:
            reward += 1.0
            done = True

        return reward, done, info


    # ------------------------------------------------------------------
    def terminou(self):
        return self._steps >= self.max_steps or any(
            pos == self.saida_pos for pos in self.posicoes_agentes.values()
        )