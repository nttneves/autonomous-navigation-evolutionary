# environments/environment_maze.py
from typing import Tuple
from environments.environment import Enviroment
import random
import numpy as np
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
                 tamanho: Tuple[int, int] = (21, 21),
                 dificuldade: int = 0,
                 max_steps: int = 200,
                 range_max: int = 5):
        super().__init__(tamanho=tamanho, dificuldade=dificuldade)

        self.max_steps = int(max_steps)
        self.saida_pos = None
        self.range_max = int(range_max)

        # criar mapa inicial
        self.mapa_estado = self.criar_mapa(dificuldade)
        self._initial_map = self.mapa_estado.copy()

        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------

    def reset(self):
        self.mapa_estado = self._initial_map.copy()
        # garante que há farol
        if self.saida_pos is None:
            self.saida_pos = self.place_exit()
        bx, by = self.saida_pos
        self.mapa_estado[by, bx] = SAIDA

        self.posicoes_agentes = {}
        self.passo_tempo = 0
        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------
    
    def place_exit(self):
        w, h = self.tamanho
        # coloca a saída na metade superior do mapa
        
        bx = random.randint(0, w - 1)
        
        return (bx,1)

    # ------------------------------------------------------------------

    def criar_mapa(self, dificuldade=0):
        w, h = self.tamanho

        # garantir impar
        if w % 2 == 0: w -= 1
        if h % 2 == 0: h -= 1

        maze = np.full((h, w), fill_value=PAREDE, dtype=int)

        # DFS
        def crave(x, y):
            directions = [(2,0), (-2,0), (0,2), (0,-2)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < w-1 and 1 <= ny < h-1 and maze[ny, nx] == PAREDE:
                    maze[ny, nx] = VAZIO
                    maze[y + dy//2, x + dx//2] = VAZIO
                    crave(nx, ny)

        # start
        maze[1,1] = VAZIO
        crave(1, 1)

        # atualizar tamanho real
        self.tamanho = (w, h)

        # saída
        self.saida_pos = self.place_exit()
        bx, by = self.saida_pos
        maze[by, bx] = SAIDA

        # spawn
        sx, sy = 0, h-1
        for dx in (0,1):
            for dy in (0,1):
                maze[sy-dy, sx+dx] = VAZIO

        # abrir caminho se saída estiver bloqueada
        neighbours = [(bx, by+1), (bx+1, by), (bx-1, by)]
        if all(0 <= nx < w and 0 <= ny < h and maze[ny,nx] == PAREDE for nx,ny in neighbours):
            maze[by+1, bx] = VAZIO

        return maze

    # ------------------------------------------------------------------

    def _in_bounds(self, x, y):
        w, h = self.tamanho
        return 0 <= x < w and 0 <= y < h

    # ------------------------------------------------------------------

    def _ray_distance(self, x0, y0, dx, dy, max_range=None):
        """
        Avança passo a passo na direcção (dx,dy) (inteiros -1,0,1)
        até encontrar uma parede ou atingir max_range.
        Retorna distância normalizada [0..1], onde 1 = sem obstáculo até max_range.
        """
        if max_range is None:
            max_range = self.range_max

        x, y = x0, y0
        for step in range(1, max_range + 1):
            nx = x + dx * step
            ny = y + dy * step
            if not self._in_bounds(nx, ny):
                return (step - 1) / float(max_range)
            if self.mapa_estado[ny, nx] == PAREDE:
                return (step - 1) / float(max_range)
        return 1.0  # sem obstáculo dentro do alcance

    # ------------------------------------------------------------------

    def _radar_quadrants(self, agente: Agent):
        """
        Divide o espaço em 4 setores centrados nas direções: front (up), right, back (down), left.
        Retorna lista 4-long com 1.0 se o farol estiver nesse sector, 0 caso contrário.
        Critério: compara abs(dx) vs abs(dy) e o sinal de dx/dy.
        """
        ax, ay = agente.posicao
        bx, by = self.saida_pos
        dx = bx - ax
        dy = by - ay

        # se a distância for zero, marca front por convenção
        if dx == 0 and dy == 0:
            return [0.0, 0.0, 0.0, 0.0]

        # decide sector
        # front: dy < 0 and abs(dy) >= abs(dx)
        # right: dx > 0 and abs(dx) > abs(dy)
        # back: dy > 0 and abs(dy) >= abs(dx)
        # left: dx < 0 and abs(dx) > abs(dy)
        if abs(dy) >= abs(dx):
            if dy < 0:
                return [1.0, 0.0, 0.0, 0.0]  # front
            else:
                return [0.0, 0.0, 1.0, 0.0]  # back
        else:
            if dx > 0:
                return [0.0, 1.0, 0.0, 0.0]  # right
            else:
                return [0.0, 0.0, 0.0, 1.0]  # left

    # ------------------------------------------------------------------

    def observacaoPara(self, agente: Agent):
        """
        Observação com 10 inputs:
        - 6 rangefinders: up, right, down, left, up-right, up-left (0..1)
        - 4 radar sectors: front, right, back, left (0/1)
        """
        ax, ay = agente.posicao

        if agente.sensores == True:
            max_r = self.range_max    
        else:
            max_r = 1                 

        # Rangefinder directions (relative to map axes)
        rf_dirs = [
            (0, -1),   # up
            (1, 0),    # right
            (0, 1),    # down
            (-1, 0),   # left
            (1, -1),   # up-right
            (-1, -1),  # up-left
        ]

        ranges = []
        for (dx, dy) in rf_dirs:
            val = self._ray_distance(ax, ay, dx, dy, max_range=max_r)
            ranges.append(float(val))

        radar = self._radar_quadrants(agente)

        return {
            "ranges": ranges,   # 6 floats normalizados
            "radar": radar      # 4 floats one-hot
        }

    # ------------------------------------------------------------------

    def atualizacao(self):
        self.passo_tempo += 1
        self._steps += 1

    # ------------------------------------------------------------------

    def agir(self, accao: int, agente: Agent):
        if agente.id not in self.posicoes_agentes:
            return 0.0, False, {}

        x, y = self.posicoes_agentes[agente.id]
        dx, dy = ACTION_TO_DELTA[accao]
        nx, ny = x + dx, y + dy

        reward = 0.0
        done = False
        info = {}

        # colisão ou fora dos limites
        if not self._in_bounds(nx, ny) or self.mapa_estado[ny, nx] == PAREDE:
            reward -= 0.01
            info["collision"] = True
            nx, ny = x, y
        else:
            self.posicoes_agentes[agente.id] = (nx, ny)
            agente.posicao = (nx, ny)

        # chegou ao farol
        if (nx, ny) == self.saida_pos:
            reward += 1.0
            done = True
            info["reached_beacon"] = True

        return reward, done, info

    # ------------------------------------------------------------------

    def terminou(self):
        if self._steps >= self.max_steps:
            return True

        for pos in self.posicoes_agentes.values():
            if pos == self.saida_pos:
                return True

        return False