import numpy as np
from sensor.sensor import Sensor

class sensor_obstaculo(Sensor):
    #Sensor que devolve a distância de obstáculos em 4 direções
    DIRECOES = [
        (0, -1),   # N
        (1, 0),    # E
        (0, 1),    # S
        (-1, 0),   # W
    ]

    def gerar_observacao(self, ambiente, agente) -> np.ndarray:
        #Devolve um vetor de 4 elementos com a distância até ao obstáculo
        x_agente, y_agente = agente.posicao
        observacao_distancia = np.full(4, self.alcance, dtype=int)
        
        for i, (dx_step, dy_step) in enumerate(self.DIRECOES):
            distancia = 0
            
            for d in range(1, self.alcance + 1):
                x_check = x_agente + (dx_step * d)
                y_check = y_agente + (dy_step * d)
                
                if ambiente.tem_obstaculo_ou_objeto_relevante(x_check, y_check):
                    observacao_distancia[i] = d
                    break
            
        return observacao_distancia