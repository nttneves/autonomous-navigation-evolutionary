# agent.py
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import queue
import json
import threading
import time

class Agent(ABC):

    def __init__(self, id: str, politica=None, sensores=None):
        self.id = id
        self.politica = politica or {}
        self.sensores = sensores or []
        self.inbox = queue.Queue()
        self.last_observation = None
        self.last_action = None
        self.rewards = []
        self.ambiente = None 

    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)
        return cls(data["id"], data.get("politica"), data.get("sensores"))
    

    @abstractmethod
    def observacao(self, obs): pass

    @abstractmethod
    def age(self): pass

    @abstractmethod
    def avaliacaoEstadoAtual(self, recompensa: float): pass


    def instala(self, sensor):
        self.sensores.append(sensor)


    def comunica(self, msg, de_agente=None):
        self.inbox.put({"from": getattr(de_agente, "id", None),
                        "message": msg,
                        "timestamp": time.time()})

    def envia(self, msg, para_agente):
        para_agente.comunica(msg, self)


    def regista_reward(self, r: float):
        self.rewards.append(r)
