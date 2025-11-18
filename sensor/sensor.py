from abc import ABC, abstractmethod

class Sensor(ABC):

    def __init__(self, alcance: int):
        self.alcance = alcance

    @abstractmethod
    def gerar_observacao(self, ambiente, agente):
        pass