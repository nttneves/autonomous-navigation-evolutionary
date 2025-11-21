#environment_farol.py
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
  
  def __init__(self, tamanho: Tuple[int, int] = (21, 21),
              dificuldade: int = 0,
              max_steps: int = 200):
    
    super().__init__(tamanho=tamanho, dificulade=dificuldade)

    # força mapa quadrado
    n = (tamanho, tamanho)

    self.dificuldade = int(dificuldade)
    self.max_steps = int(max_steps)

    # posição do farol será criada no mapa
    self.farol_pos = None

    # gerar mapa procedural
    self.mapa_estado = self.criar_mapa(dificuldade)
    self._initial_map = self.mapa_estado.copy()

    self._steps = 0
    self._done = False
    
  def reset(self):
    self.mapa_estado = self._initial_map.copy()
    bx, by = self.farol_pos
    self.mapa_estado[by, bx] = FAROL
    self.posicoes_agentes = {}
    self.passo_tempo = 0
    self._steps = 0
    self._done = False
    
  def place_beacon(self):
    ymax = max(0, self.tamanho // 2 - 1)
    bx = random.randint(0, self.tamanho - 1)
    by = random.randint(0, ymax)
    return (bx, by)
  
  def criar_mapa(self, dificulade=0):
    mapa = np.full(self.n, fill_value=VAZIO, dtype=int)
    
    self.farol_pos = self.place_beacon()
    beacon = self.farol_pos
    
    bx, by = beacon
    mapa[bx, by] = FAROL
    return mapa
  
  def posicao_relativa_farol(self, agente: Agent):
    ax, ay = agente.posicao
    bx, by = self.farol_pos
      
    dx = bx - ax
    dy = by - ay
      
    dist = np.sqrt(dx**2 + dy**2)
    direction = (dx / dist, dy/ dist)
    return direction
  
  
  def observacaoPara(self, agente: Agent):
    obs = {}
    dx, dy =  self.posicao_relativa_farol(agente)
    ax, ay = agente.posicao
    
    # TODO: FALTA AQUI UMA VALIDAÇÃO CASO O AGENTE ESTEJA NAS BORDAS DO MAPA ALGUMAS COORDENADAS SÃO INVÁLIDAS
    #NÃO SEI COMO É QUE O PYTHON FUNCIONA EM RELAÇÃO A ISTO
    if agente.sensores == True:
      obs = {self.mapa_estado[ax-1, ay], self.mapa_estado[ax, ay-1], self.mapa_estado[ax+1, ay], self.mapa_estado[ax, ay-1], 
             self.mapa_estado[ax-1, ay+1], self.mapa_estado[ax+1, ay+1], self.mapa_estado[ax-1, ay-1], self.mapa_estado[ax+1, ay-1],
             dx, dy}
    elif agente.sensores == False:
       obs = {self.mapa_estado[ax-1, ay], self.mapa_estado[ax, ay-1], self.mapa_estado[ax+1, ay], self.mapa_estado[ax, ay-1], 
             0, 0, 0, 0, dx, dy}
    
    return obs
  
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
  
  def _in_bounds(self, x, y):
    w, h = self.tamanho
    return 0 <= x < w and 0 <= y < h
    
  
  def terminou(self):
    if self._steps >= self.max_steps:
      return True

    for pos in self.posicoes_agentes.values():
      if pos == self.farol_pos:
        return True

    return False
  
  