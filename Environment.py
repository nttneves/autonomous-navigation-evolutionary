import numpy as np
from abc import ABC, abstractmethod

PAREDE = -1
VAZIO = 0
RECURSO = 1
NINHO = 2
AGENTE = 3

class Enviroment(ABC):
    def __init__(self, mapa_file: str, tamanho: tuple):
        self.tamanho = tamanho  
        self.mapa_estado = self._carregar_mapa(mapa_file)

        self.agentes = {}
        self.posicoes_agentes = {}
        self.passo_tempo = 0

    @abstractmethod
    def _carregar_mapa(self, mapa_file: str) -> np.ndarray:
        pass

    @abstractmethod
    def atualizacao(self):
        """Chamado APÓS todos os agentes agirem."""
        self.passo_tempo += 1
        pass

    @abstractmethod
    def agir(self, accao: str, agente):
        self.posicoes_agentes[agente.id] = (nx, ny)
        agente.posicao = (nx, ny)
    
    def regista_agente(self, agente, pos_inicial: tuple):
        self.agentes[agente.id] = agente
        self.posicoes_agentes[agente.id] = pos_inicial
        try:
            agente.posicao = tuple(pos_inicial)
        except Exception:
            pass

    def get_posicao_agente(self, agente):
        return self.posicoes_agentes.get(agente.id)


    
    def observacaoPara(self, agente) -> dict:
        observacao = {}
        for nome, sensor in agente.sensores.items():
            observacao[nome] = sensor.gerar_observacao(self, agente)
        return observacao

    # ----------------------------------------------------------------------
    # MÉTODOS AUXILIARES USADOS PELOS SENSORES

    def tem_obstaculo(self, x: int, y: int) -> bool:
        """Um obstáculo é PAREDE OU fora dos limites."""
        if not (0 <= x < self.tamanho[0] and 0 <= y < self.tamanho[1]):
            return True 
        return self.mapa_estado[y, x] == PAREDE

    def tem_obstaculo_ou_objeto_relevante(self, x: int, y: int) -> bool:
        """
        Usado pelos sensores de objectos e obstaculos.
        Pode ser personalizado por subclasse.
        """
        if not (0 <= x < self.tamanho[0] and 0 <= y < self.tamanho[1]):
            return True
        
        cel = self.mapa_estado[y, x]

        # Considera obstáculo:
        if cel == PAREDE:
            return True

        # Considera recurso ou objetos como obstacles relevantes:
        if cel in [RECURSO, NINHO]:
            return True

        return False

    def get_submatriz(self, x_min, x_max, y_min, y_max):
        return self.mapa_estado[y_min:y_max, x_min:x_max].copy()
