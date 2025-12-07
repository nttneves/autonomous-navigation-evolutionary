# environment_maze.py
from environments.environment import Enviroment, VAZIO, PAREDE, GOAL
import numpy as np
import random

class MazeEnv(Enviroment):

    def __init__(self, dificuldade=0, max_steps: int = 200):
        if dificuldade == 0:
            tamanho = (12,12); seed = 42
        elif dificuldade == 1:
            tamanho = (20,20); seed = 150
        else:
            tamanho = (30,30); seed = 456

        self.seed = seed
        super().__init__(tamanho, dificuldade, max_steps)

        self.mapa_estado, self.goal_pos = self.criar_mapa()
        self._initial_map = self.mapa_estado.copy()

    def criar_mapa(self):
        w, h = self.tamanho
        rng = random.Random(self.seed)

        maze = np.full((h, w), PAREDE, dtype=int)

        def carve(x,y):
            dirs = [(2,0),(-2,0),(0,2),(0,-2)]
            rng.shuffle(dirs)
            for dx,dy in dirs:
                nx, ny = x+dx, y+dy
                if 1 <= nx < w-1 and 1 <= ny < h-1 and maze[ny, nx] == PAREDE:
                    maze[ny, nx] = VAZIO
                    maze[y+dy//2, x+dx//2] = VAZIO
                    carve(nx, ny)

        maze[1,1] = VAZIO
        carve(1,1)

        exits = [x for x in range(1,w-1) if maze[1,x] == VAZIO]
        sx = rng.choice(exits) if exits else w//2

        goal_pos = (sx, 0)
        maze[0, sx] = GOAL

        maze[h-1,1] = maze[h-2,1] = maze[h-1,2] = VAZIO

        return maze, goal_pos