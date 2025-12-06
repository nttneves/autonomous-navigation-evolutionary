# environment_maze.py
from environments.environment import Enviroment, VAZIO, PAREDE, GOAL
import numpy as np
import random

class MazeEnv(Enviroment):

    def __init__(self, dificuldade=0, max_steps: int = 200):
        #"só 2 dificuldades"
        if dificuldade == 0:
            tamanho = (21,21); seed = 24
        elif dificuldade == 1:
            tamanho = (31,31); seed = 187
        

        self.seed = seed
        super().__init__(tamanho, dificuldade, max_steps)

        
    

        self.mapa_estado = self.criar_mapa(linhas=dificuldade*6 + 6)
        self._initial_map = self.mapa_estado.copy()
        # definir objetivo
        self.goal_pos = self.saida_pos

    def criar_mapa(self,linhas):
        w,h = self.tamanho
        maze = np.zeros((h, w), dtype=int)
        # Gerador próprio com seed
        rng = random.Random(self.seed)

        
        # Bordas do labirinto
        maze[0, :] = PAREDE
        maze[h-1, :] = PAREDE
        maze[:, 0] = PAREDE
        maze[:, w-1] = PAREDE

        # Função para desenhar linhas (paredes) diagonais
        def linha(x1, y1, x2, y2):
            dx = abs(x2 - x1)
            dy = -abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx + dy
            
            prev_x, prev_y = x1, y1  # para não contar a célula anterior
            primeiro_ponto = 0

            while True:
                # Distância mínima da borda
                distancia_borda = 2
                perto_borda = x1 < distancia_borda or x1 >= w - distancia_borda or y1 < distancia_borda or y1 >= h - distancia_borda

                if primeiro_ponto >= 2 and not perto_borda:
                    # checagem 3x3 após os primeiros dois pontos e longe da borda
                    safe = True
                    for ny in range(y1-1, y1+2):
                        for nx in range(x1-1, x1+2):
                            if 0 <= nx < w and 0 <= ny < h:
                                if (nx, ny) != (prev_x, prev_y) and maze[ny, nx] == PAREDE:
                                    safe = False
                    if safe:
                        maze[y1, x1] = PAREDE
                        prev_x, prev_y = x1, y1
                else:
                    # primeiros dois pontos ou perto da borda: sempre marcar
                    maze[y1, x1] = PAREDE
                    prev_x, prev_y = x1, y1
                    primeiro_ponto += 1

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x1 += sx
                if e2 <= dx:
                    err += dx
                    y1 += sy

        # Exemplo de paredes diagonais
        # Número de linhas que quer gerar
        num_linhas = linhas

        # Geração de linhas conectadas a paredes existentes
        for _ in range(num_linhas):
            # Escolhe um ponto inicial aleatório de uma parede existente
            paredes_existentes = [(x, y) for y in range(h) for x in range(w) if maze[y, x] == PAREDE]
            x1, y1 = rng.choice(paredes_existentes)

            # Escolhe um ponto final aleatório dentro do labirinto
            x2 = rng.randint(0, w-1)
            y2 = rng.randint(0, h-1)

            linha(x1, y1, x2, y2)

    

        # Entrada na parte inferior esquerda
        maze[h-2,1] = VAZIO
        maze[h-1,1] = VAZIO
        #maze[h-1,0] = VAZIO

        # Escolher posição da saída no topo
        sx = rng.randint(1, w-2)   # evita as bordas
        self.saida_pos = (sx, 0)
        maze[0, sx] = GOAL
        

        return maze








     