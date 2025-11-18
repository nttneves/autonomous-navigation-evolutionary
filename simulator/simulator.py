# simulator/simulator.py
import numpy as np

class Simulator:
    """
    Invólucro (wrapper) simples e síncrono para um simulador de agente único.
    """

    def __init__(self, env, max_steps=None, agent=None):
        self.env = env
        self.max_steps = int(max_steps) if max_steps is not None else env.max_steps
        self.agent = agent
        self._last_obs = None
        self._done = True
        self._steps = 0
        self._total_reward = 0.0

    def run_episode(self, agent, render=False):
        obs = self.env.reset()
        traj = [tuple(self.env.agent_pos.copy())]
        total_reward = 0.0
        done = False
        steps = 0
        # lopp do episódio- continua até terminar ou atingir o máximo de passos
        while not done and steps < self.max_steps:
            # O agente escolhe uma ação com base na observação atual
            a = agent.act(obs)
            # aplica a ação no ambiente
            obs, r, done, info = self.env.step(a)
            total_reward += r
            traj.append(tuple(self.env.agent_pos.copy()))
            steps += 1
        # posição final do agente
        final = tuple(self.env.agent_pos.copy())
        # posição do objetivo
        goal = tuple(self.env.goal_pos.copy()) if hasattr(self.env, "goal_pos") else None
        # agente alcançou o objetivo?
        reached = (final == goal) if goal is not None else False

        # Retorna um dicionário com os resultados do episódio
        return {
            "total_reward": float(total_reward),
            "steps": int(steps),
            "traj": traj,  # Trajetória (lista de posições)
            "final_pos": final,
            "goal_pos": goal,
            "reached_goal": bool(reached)
        }

    def attach_agent(self, agent):
        # Liga (associa) um agente ao simulador para uso na API de passos visuais
        self.agent = agent

    def reset_env(self):
        # Reinicia o ambiente e o estado interno do simulador
        self._last_obs = self.env.reset()
        self._done = False
        self._steps = 0
        self._total_reward = 0.0
        return self._last_obs

    def step_once_for_visualiser(self, action=None):
        """
        Avança um único "tick" (passo). 
        Se a 'action' for None, usa o agente associado para escolher a ação a partir da observação atual.
        Retorna (obs, recompensa, terminou, info)
        """
        # Se o episódio anterior tiver terminado, reinicia o ambiente
        if self._done:
            self.reset_env()
        # Obtém a observação atual
        obs = self._last_obs
        
        if action is None:
            # Se a ação não foi fornecida, usa o agente
            if self.agent is None:
                raise RuntimeError("No agent attached for automatic stepping")
            a = self.agent.act(obs)
        else:
            a = int(action)
            
        # Executa o passo no ambiente
        obs2, reward, done, info = self.env.step(a)
        
        # Atualiza o estado interno
        self._last_obs = obs2
        self._done = bool(done)
        self._steps += 1
        self._total_reward += reward
        
        return obs2, float(reward), bool(done), info or {}