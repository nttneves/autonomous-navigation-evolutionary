# environment_farol.py
import numpy as np
import random
from typing import Tuple, Optional
from collections import deque
from copy import deepcopy

try:
    from environments.environment import Enviroment, PAREDE, VAZIO, RECURSO, NINHO, AGENTE
except Exception:
    from environment import Enviroment
    PAREDE = -1
    VAZIO = 0
    RECURSO = 1
    NINHO = 2
    AGENTE = 3


ACTION_TO_DELTA = {
    0: (0, -1),   # up
    1: (1, 0),    # right
    2: (0, 1),    # down
    3: (-1, 0),   # left
}


class FarolEnv(Enviroment):
    """
    Ambiente Farol:
      - mapa quadrado
      - dificuldade 0/1/2
      - garante sempre solução
      - agente começa no canto inferior esquerdo (0, n-1)
      - farol na metade superior
      - O AMBIENTE NÃO CRIA AGENTES, NÃO INSTALA SENSORES, NÃO REGISTA AGENTES
    """

    def __init__(self,
                 tamanho: Tuple[int, int] = (21, 21),
                 dificuldade: int = 0,
                 max_steps: int = 200):

        # força mapa quadrado
        n = min(tamanho[0], tamanho[1])
        tamanho = (n, n)

        self.dificuldade = int(dificuldade)
        self.max_steps = int(max_steps)

        # posição do farol será criada no mapa
        self.farol_pos = None

        super().__init__(mapa_file=None, tamanho=tamanho)

        # gerar mapa procedural
        self.mapa_estado = self._carregar_mapa()
        self._initial_map = self.mapa_estado.copy()

        self._steps = 0
        self._done = False

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _start_pos(self):
        w, h = self.tamanho
        return (0, h - 1)

    def _place_beacon(self):
        w, h = self.tamanho
        # farol na metade superior
        ymax = max(0, h // 2 - 1)
        bx = random.randint(0, w - 1)
        by = random.randint(0, ymax)
        return (bx, by)

    def _in_bounds(self, x, y):
        w, h = self.tamanho
        return 0 <= x < w and 0 <= y < h

    def _neighbours(self, x, y):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
                yield (nx, ny)

    def _bfs_reachable(self, mapa, start, goal):
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return True
            for nx, ny in self._neighbours(x, y):
                if (nx, ny) not in visited and mapa[ny, nx] != PAREDE:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _carve_path(self, mapa, start, goal):
        """
        Gera caminho serpenteado estilo DFS randomizado
        similar ao visual da imagem que mostraste.
        """
        path = []
        stack = [start]
        visited = {start}

        while stack:
            x, y = stack[-1]
            path.append((x, y))
            if (x, y) == goal:
                break

            neighbors = list(self._neighbours(x, y))
            random.shuffle(neighbors)
            moved = False

            for nx, ny in neighbors:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    moved = True
                    break

            if not moved:
                stack.pop()

        # limpar caminho no mapa
        for x, y in path:
            mapa[y, x] = VAZIO

        return set(path)

    def _add_random_walls(self, mapa, reserved, density):
        total = mapa.size
        num = int(total * density)
        w, h = self.tamanho

        placed = 0
        attempts = 0
        max_attempts = num * 10

        start = self._start_pos()
        goal = self.farol_pos

        while placed < num and attempts < max_attempts:
            attempts += 1

            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            if (x, y) in reserved:
                continue

            if mapa[y, x] == PAREDE:
                continue

            # tenta colocar parede
            mapa[y, x] = PAREDE

            # se quebra o caminho, desfaz
            if not self._bfs_reachable(mapa, start, goal):
                mapa[y, x] = VAZIO
            else:
                placed += 1

    # ------------------------------------------------------------
    # Mapa procedural (núcleo)
    # ------------------------------------------------------------
    def _carregar_mapa(self, mapa_file=None):
        w, h = self.tamanho
        mapa = np.full((h, w), VAZIO, dtype=int)

        start = self._start_pos()
        self.farol_pos = self._place_beacon()
        beacon = self.farol_pos

        # caminho garantido
        reserved = self._carve_path(mapa, start, beacon)

        # paredes extra conforme dificuldade
        if self.dificuldade == 1:
            self._add_random_walls(mapa, reserved, density=0.10)
        elif self.dificuldade == 2:
            self._add_random_walls(mapa, reserved, density=0.25)

        # marca farol
        bx, by = beacon
        mapa[by, bx] = RECURSO

        return mapa

    # ------------------------------------------------------------
    # Métodos principais do Ambiente
    # ------------------------------------------------------------
    def reset(self):
        self.mapa_estado = self._initial_map.copy()
        bx, by = self.farol_pos
        self.mapa_estado[by, bx] = RECURSO
        self.posicoes_agentes = {}
        self.passo_tempo = 0
        self._steps = 0
        self._done = False

    def atualizacao(self):
        self.passo_tempo += 1
        self._steps += 1

    def agir(self, accao: int, agente):
        if agente.id not in self.posicoes_agentes:
            # simulador é responsável por registar agentes
            return 0.0, False, {}

        x, y = self.posicoes_agentes[agente.id]
        dx, dy = ACTION_TO_DELTA[accao]
        nx, ny = x + dx, y + dy

        reward = 0.0
        done = False
        info = {}

        # colisão
        if not self._in_bounds(nx, ny) or self.mapa_estado[ny, nx] == PAREDE:
            reward -= 0.01
            nx, ny = x, y
            info["collision"] = True
        else:
            self.posicoes_agentes[agente.id] = (nx, ny)

        # atingiu o farol
        if (nx, ny) == self.farol_pos:
            reward += 1.0
            done = True
            info["reached_beacon"] = True

        return reward, done, info

    def terminou(self):
        if self._steps >= self.max_steps:
            return True

        for pos in self.posicoes_agentes.values():
            if pos == self.farol_pos:
                return True

        return False

    # opcional para debug
    def desenha_mapa(self, figsize=(6,6)):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.imshow(self.mapa_estado, origin="upper", cmap="binary")
        plt.title(f"FarolEnv (dif={self.dificuldade})")
        plt.show()