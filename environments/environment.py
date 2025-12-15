from abc import ABC, abstractmethod
import numpy as np
from agents.agent import Agent
import math

VAZIO = 0
PAREDE = 1
GOAL  = 2   

ACTION_TO_DELTA = {
    0: (0, -1),   # up
    1: (1, 0),    # right
    2: (0, 1),    # down
    3: (-1, 0),   # left
}

class Enviroment(ABC):

    def __init__(self, tamanho: tuple, dificuldade: int, max_steps: int = 200):
        self.tamanho = tamanho
        self.dificuldade = dificuldade
        self.mapa_estado = None
        self.goal_pos = None

        self.agentes = {}
        self.posicoes_agentes = {}

        self._steps = 0
        self.max_steps = max_steps

    # ================================================================
    # ABSTRACTS: a única coisa que muda entre ambientes
    # ================================================================

    @abstractmethod
    def criar_mapa(self) -> tuple[np.ndarray, tuple]:
        """
        Deve devolver:
            mapa_estado (np.ndarray)
            goal_pos    (x,y)
        """
        pass

    # ================================================================
    # Métodos comuns aos ambientes
    # ================================================================

    def reset(self):
        """Repõe o estado inicial do ambiente."""
        if hasattr(self, "_initial_map"):
            self.mapa_estado = self._initial_map.copy()

        self.posicoes_agentes = {}
        self._steps = 0

        gx, gy = self.goal_pos
        self.mapa_estado[gy, gx] = GOAL

    # -------------------------------------------------------------

    def regista_agente(self, agente: Agent, pos_inicial: tuple):
        self.agentes[agente.id] = agente
        self.posicoes_agentes[agente.id] = pos_inicial
        agente.posicao = pos_inicial

    def get_posicao_agente(self, agente: Agent):
        return self.posicoes_agentes.get(agente.id)

    # ================================================================
    # Sensores genéricos (iguais para farol e maze)
    # ================================================================

    def _in_bounds(self, x, y):
        w, h = self.tamanho
        return 0 <= x < w and 0 <= y < h

    def _ray_distance(self, x0, y0, dx, dy, max_range):
        for step in range(1, max_range + 1):
            nx = x0 + dx * step
            ny = y0 + dy * step
            if not self._in_bounds(nx, ny):
                return (step - 1) / max_range
            if self.mapa_estado[ny, nx] == PAREDE:
                return (step - 1) / max_range
        return 1.0

    def _radar_quadrants(self, agente: Agent):
        ax, ay = agente.posicao
        bx, by = self.goal_pos

        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:
            return [0, 0, 0, 0]

        if abs(dy) >= abs(dx):
            return [1,0,0,0] if dy < 0 else [0,0,1,0]
        else:
            return [0,1,0,0] if dx > 0 else [0,0,0,1]

    def observacaoPara(self, agente: Agent):
        pass

    # ================================================================
    # Movimento genérico (igual Maze/Farol)
    # ================================================================

    def agir(self, accao: int, agente: Agent):
        x, y = self.posicoes_agentes[agente.id]
        dx, dy = ACTION_TO_DELTA[accao]
        nx, ny = x + dx, y + dy

        #reward = 0.0
        done = False
        inff={}

        if not self._in_bounds(nx, ny) or self.mapa_estado[ny, nx] == PAREDE:
            #reward -= 0.01
            inff["collision"] = True
            nx, ny = x, y
        else:
            self.posicoes_agentes[agente.id] = (nx, ny)
            agente.posicao = (nx, ny)

        if (nx, ny) == self.goal_pos:
            #reward += 1.0
            inff["reached_beacon"] = True
            done = True

        #return reward, done, {}
        return 0.0, done, inff

    # ================================================================
    # Fim de episódio
    # ================================================================

    def terminou(self):
        if self._steps >= self.max_steps:
            return True
        return any(pos == self.goal_pos for pos in self.posicoes_agentes.values())

    def atualizacao(self):
        self._steps += 1