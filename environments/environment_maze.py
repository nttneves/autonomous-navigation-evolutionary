# environment_maze.py
from environments.environment import Enviroment, VAZIO, PAREDE, GOAL
import numpy as np
import random
import agents.agent as Agent
import math

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
    

    def observacaoPara(self, agente: Agent):
        ax, ay = agente.posicao
        bx, by = self.goal_pos
        w, h = self.tamanho
        max_r = agente.sensors_range if hasattr(agente, "sensors_range") else 5

        rf_dirs = [
            (0, -1),  # 1. N
            (1, 0),   # 2. E
            (0, 1),   # 3. S
            (-1, 0),  # 4. O
            (1, -1),  # 5. NE
            (1, 1),   # 6. SE
            (-1, 1),  # 7. SO
            (-1, -1), # 8. NO 
        ]

        ranges = [self._ray_distance(ax, ay, dx, dy, max_r) for dx, dy in rf_dirs]

        # a) Posição Normalizada (2 valores)
        pos_x_norm = ax / max(1, w - 1)
        pos_y_norm = ay / max(1, h - 1)
        # b) Distância ao Objetivo Normalizada (1 valor)
        euclid_dist = math.hypot(ax - bx, ay - by)
        max_map_dist = math.hypot(w - 1, h - 1)
        dist_norm = euclid_dist / max(1.0, max_map_dist)
        # c) Diferença de Coordenadas Y Normalizada (1 valor)
        # (Usado como proxy para a direção vertical do objetivo)
        delta_y_norm = (by - ay) / max(1, h-1)
        #delta_x_norm = (bx - ax) / max(1, w)

        observation_vector = np.concatenate([
            np.array(ranges, dtype=np.float32),      # 8 valores
            np.array([pos_x_norm], dtype=np.float32),# 1 valor
            np.array([pos_y_norm], dtype=np.float32),# 1 valor
            np.array([dist_norm], dtype=np.float32), # 1 valor
            np.array([delta_y_norm], dtype=np.float32), # 1 valor
           
        ])

        return observation_vector
