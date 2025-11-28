# environment_farol.py
from environments.environment import Enviroment, VAZIO, PAREDE, GOAL
import random
import numpy as np

class FarolEnv(Enviroment):

    def __init__(self, tamanho=(21,21), dificuldade=0, max_steps: int = 200):
        self.range_max = 5
        super().__init__(tamanho, dificuldade, max_steps)

        self.mapa_estado, self.goal_pos = self.criar_mapa()
        self._initial_map = self.mapa_estado.copy()

    def criar_mapa(self):
        w, h = self.tamanho
        mapa = np.full((h,w), VAZIO, dtype=int)

        # farol (goal)
        bx = random.randint(0, w-1)
        by = random.randint(0, h//3)
        goal_pos = (bx,by)
        mapa[by,bx] = GOAL

        paredes = int((w*h) * 0.02 * self.dificuldade)

        for _ in range(paredes):
            x = random.randint(1, w-2)
            y = random.randint(1, h-2)
            if (x,y) != goal_pos:
                mapa[y,x] = PAREDE

        return mapa, goal_pos