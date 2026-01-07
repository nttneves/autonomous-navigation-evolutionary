# environment_farol.py
from environments.environment import Enviroment, VAZIO, PAREDE, GOAL
import random
import numpy as np
import agents.agent as Agent
import math


class FarolEnv(Enviroment):

    def __init__(self, tamanho=(21,21), dificuldade=0, max_steps: int = 200, seed=None):
        self.range_max = 5

        # guardar seed (se None → ambiente aleatório)
        self.seed = seed if seed is not None else random.randint(0, 999999)

        super().__init__(tamanho, dificuldade, max_steps)

        # gerar mapa determinístico
        self.mapa_estado, self.goal_pos = self.criar_mapa()

        # guardar cópia para resets
        self._initial_map = self.mapa_estado.copy()


    # ============================================================
    # Criar Mapa com Seed Fixa
    # ============================================================
    def criar_mapa(self):
        w, h = self.tamanho

        rng = random.Random(self.seed)

        mapa = np.full((h, w), VAZIO, dtype=int)

        # --------------------------------------------------------
        # 1. Farol não pode estar nas bordas
        #    Está sempre na metade superior
        # --------------------------------------------------------
        bx = rng.randint(1, w - 2)        # evita bordas esquerda/direita
        by = rng.randint(1, h // 3)       # evita bordas superior, fica na parte de cima
        goal_pos = (bx, by)
        mapa[by, bx] = GOAL

        # --------------------------------------------------------
        # 2. Paredes (densidade depende da dificuldade)
        # --------------------------------------------------------
        densidade = min(0.02 * self.dificuldade, 0.25)
        num_paredes = int((w * h) * densidade)

        for _ in range(num_paredes):
            x = rng.randint(1, w - 2)
            y = rng.randint(1, h - 2)

            # nunca colocar paredes em cima do goal
            if (x, y) == goal_pos:
                continue

            mapa[y, x] = PAREDE

        return mapa, goal_pos
    
    def compute_reward(self, agent, prev_pos, new_pos, info):
        reward = -5.0
        done = False

        bx, by = self.goal_pos
        px, py = prev_pos
        nx, ny = new_pos

        prev_dist = math.hypot(px - bx, py - by)
        new_dist  = math.hypot(nx - bx, ny - by)
        delta = prev_dist - new_dist

        if info.get("collision", False):
            reward -= 5.0
        else:
            reward += delta * (10.0 if delta > 0 else 50.0)

        if new_pos == self.goal_pos:
            reward += 1000.0 + (self.max_steps - self._steps) * 2.0
            done = True

        return reward, done
    

    def observacaoPara(self, agente: Agent):
        ax, ay = agente.posicao
        max_r = self.range_max

        # --------------------------------------------------
        # 1) Ray sensors (paredes) — 8 direções
        # --------------------------------------------------
        rf_dirs = [
            (0, -1),   # N
            (1, 0),    # E
            (0, 1),    # S
            (-1, 0),   # W
            (1, -1),   # NE
            (1, 1),    # SE
            (-1, 1),   # SW
            (-1, -1),  # NW
        ]

        ranges = [
            self._ray_distance(ax, ay, dx, dy, max_r)
            for dx, dy in rf_dirs
        ]

        ranges = np.array(ranges, dtype=np.float32)  # 8 valores

        # --------------------------------------------------
        # 2) Radar direcional do farol — 4 quadrantes
        # --------------------------------------------------
        radar = np.array(
            self._radar_quadrants(agente),
            dtype=np.float32
        )  # 4 valores

        # --------------------------------------------------
        # 3) Observação final
        # --------------------------------------------------
        observation = np.concatenate([ranges, radar])

        return observation