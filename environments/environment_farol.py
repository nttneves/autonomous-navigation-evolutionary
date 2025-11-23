# environment_farol.py
from typing import Tuple
from environments.environment import Enviroment
import random
import numpy as np
from agents.agent import Agent

ACTION_TO_DELTA = {
    0: (0, -1),   # up
    1: (1, 0),    # right
    2: (0, 1),    # down
    3: (-1, 0),   # left
}

VAZIO = 0
PAREDE = 1
FAROL = 2

class FarolEnv(Enviroment):

    def __init__(self,
                 tamanho: Tuple[int, int] = (21, 21),
                 dificuldade: int = 0,
                 max_steps: int = 200):

        super().__init__(tamanho=tamanho, dificuldade=dificuldade)

        self.max_steps = int(max_steps)
        self.farol_pos = None

        # criar mapa inicial
        self.mapa_estado = self.criar_mapa(dificuldade)
        self._initial_map = self.mapa_estado.copy()

        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------

    # def reset(self):
    #     self.mapa_estado = self._initial_map.copy()
    #     bx, by = self.farol_pos
    #     self.mapa_estado[by, bx] = FAROL

    #     self.posicoes_agentes = {}
    #     self.passo_tempo = 0
    #     self._steps = 0
    #     self._done = False

    def reset(self):
        """
        Reinicia o ambiente e coloca o farol numa nova posição aleatória.
        O mapa base é recriado sempre que o episódio recomeça.
        """

        # novo farol
        self.farol_pos = self.place_beacon()

        # recriar mapa inicial com novo farol
        w, h = self.tamanho
        self.mapa_estado = np.full((h, w), fill_value=VAZIO, dtype=int)
        bx, by = self.farol_pos
        self.mapa_estado[by, bx] = FAROL

        # guardar como mapa inicial deste episódio
        self._initial_map = self.mapa_estado.copy()

        # limpar posições dos agentes
        self.posicoes_agentes = {}
        self.passo_tempo = 0
        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------

    def place_beacon(self):
        w, h = self.tamanho
        # coloca o farol na metade superior do mapa
        ymax = max(0, h // 2 - 1)
        bx = random.randint(1, w - 1)
        by = random.randint(0, ymax)
        return (bx, by)

    # ------------------------------------------------------------------

    def criar_mapa(self, dificuldade=0):
        w, h = self.tamanho
        mapa = np.full((h, w), fill_value=VAZIO, dtype=int)

        self.farol_pos = self.place_beacon()
        bx, by = self.farol_pos
        mapa[by, bx] = FAROL

        # dificuldade futura: paredes aleatórias
        return mapa

    # ------------------------------------------------------------------

    def posicao_relativa_farol(self, agente: Agent):
        ax, ay = agente.posicao
        bx, by = self.farol_pos

        dx = bx - ax
        dy = by - ay

        dist = np.sqrt(dx * dx + dy * dy)
        if dist == 0:
            return (0.0, 0.0)

        return (dx / dist, dy / dist)

    # ------------------------------------------------------------------

    def _safe_read(self, x, y):
        """Devolve conteúdo do mapa ou parede se for fora dos limites."""
        if 0 <= x < self.tamanho[0] and 0 <= y < self.tamanho[1]:
            return self.mapa_estado[y, x]
        return PAREDE

    # ------------------------------------------------------------------

    def observacaoPara(self, agente: Agent):
      ax, ay = agente.posicao
      dx, dy = self.posicao_relativa_farol(agente)

      # 8 direções em ordem consistente (importante para rede neuronal)
      leituras = [
          (ax,     ay - 1),  # N
          (ax + 1, ay),      # E
          (ax,     ay + 1),  # S
          (ax - 1, ay),      # W
          (ax + 1, ay - 1),  # NE
          (ax + 1, ay + 1),  # SE
          (ax - 1, ay + 1),  # SW
          (ax - 1, ay - 1),  # NW
      ]

      viz = []

      if agente.sensores:
          # vê tudo (8 direções)
          for (x, y) in leituras:
              viz.append(self._safe_read(x, y))
      else:
          # vê só 4 direções (N,E,S,W)
          for i, (x, y) in enumerate(leituras):
              if i < 4:
                  viz.append(self._safe_read(x, y))
              else:
                  viz.append(-1)   # diagonais invisíveis

      # formato final
      return {
          "sensores": viz,       # lista de 8 números
          "dir_farol": (dx, dy)  # tupla de 2 floats normalizados
      }

    # ------------------------------------------------------------------

    def atualizacao(self):
        self.passo_tempo += 1
        self._steps += 1

    # ------------------------------------------------------------------

    # def agir(self, accao: int, agente: Agent):
    #     if agente.id not in self.posicoes_agentes:
    #         return 0.0, False, {}

    #     x, y = self.posicoes_agentes[agente.id]
    #     dx, dy = ACTION_TO_DELTA[accao]
    #     nx, ny = x + dx, y + dy

    #     reward = 0.0
    #     done = False
    #     info = {}

    #     # colisão ou fora dos limites
    #     if not self._in_bounds(nx, ny) or self.mapa_estado[ny, nx] == PAREDE:
    #         reward -= 0.01
    #         info["collision"] = True
    #         nx, ny = x, y
    #     else:
    #         self.posicoes_agentes[agente.id] = (nx, ny)

    #     # chegou ao farol
    #     if (nx, ny) == self.farol_pos:
    #         reward += 1.0
    #         done = True
    #         info["reached_beacon"] = True

    #     return reward, done, info

    def agir(self, accao: int, agente: Agent):
        if agente.id not in self.posicoes_agentes:
            return 0.0, False, {}

        x, y = self.posicoes_agentes[agente.id]
        dx, dy = ACTION_TO_DELTA[accao]
        nx, ny = x + dx, y + dy

        reward = 0.0
        done = False
        info = {}

        # distância ao farol antes do movimento
        bx, by = self.farol_pos
        dist_before = np.sqrt((bx - x) ** 2 + (by - y) ** 2)

        # colisão ou fora dos limites
        if not self._in_bounds(nx, ny) or self.mapa_estado[ny, nx] == PAREDE:
            reward -= 0.01
            info["collision"] = True
            nx, ny = x, y
        else:
            self.posicoes_agentes[agente.id] = (nx, ny)

        # distância depois do movimento
        dist_after = np.sqrt((bx - nx) ** 2 + (by - ny) ** 2)

        # reward shaping: ganho por aproximação (positivo se se aproximou)
        # aumento do factor de escala para dar sinal mais forte
        reward_scale = 0.5
        reward += (dist_before - dist_after) * reward_scale

        # pequeno incentivo para subir se o farol estiver acima
        if by < y and accao == 0:
            reward += 0.05

        # chegar ao farol dá recompensa grande
        if (nx, ny) == self.farol_pos:
            reward += 1.0
            done = True
            info["reached_beacon"] = True

        return reward, done, info

    # ------------------------------------------------------------------

    def _in_bounds(self, x, y):
        w, h = self.tamanho
        return 0 <= x < w and 0 <= y < h

    # ------------------------------------------------------------------

    def terminou(self):
        if self._steps >= self.max_steps:
            return True

        for pos in self.posicoes_agentes.values():
            if pos == self.farol_pos:
                return True

        return False