# simulator/simulator.py
import numpy as np

class Simulator:
    """
    Simulador responsável por:
      - resetar o ambiente
      - registar o agente na posição inicial
      - ciclo percepção → deliberação → ação
    """

    def __init__(self, env, max_steps=None):
        self.env = env
        self.max_steps = int(max_steps) if max_steps is not None else env.max_steps

    # ---------------------------------------------------------
    # EPISÓDIO COMPLETO
    # ---------------------------------------------------------
    def run_episode(self, agent, render=False):
        """
        Corre um único episódio seguindo exatamente o diagrama:
          1. env.reset()
          2. registar agente
          3. loop de:
                observacao → agente.age() → env.agir()
                env.atualizacao()
        """
        # 1) Reset ambiente
        self.env.reset()

        # 2) Posicionar o agente no canto inferior esquerdo
        start_pos = (0, self.env.tamanho[1] - 1)
        self.env.regista_agente(agent, start_pos)

        # 3) Loop principal
        traj = [start_pos]
        total_reward = 0.0
        steps = 0
        done = False

        # primeira observação
        obs = self.env.observacaoPara(agent)

        while not done and steps < self.max_steps:
            # 3.1) agente delibera ação
            action = agent.age(obs)

            # 3.2) ambiente reage
            reward, done, info = self.env.agir(action, agent)
            total_reward += reward

            # 3.3) observar novo estado
            obs = self.env.observacaoPara(agent)

            # 3.4) registrar posição
            pos = self.env.get_posicao_agente(agent)
            traj.append(pos)

            # 3.5) ambiente avança o tempo
            if hasattr(self.env, "atualizacao"):
                self.env.atualizacao()

            steps += 1

        # Posição final
        final_pos = self.env.get_posicao_agente(agent)
        goal_pos = getattr(self.env, "farol_pos", None)
        reached = (goal_pos is not None and final_pos == goal_pos)

        return {
            "total_reward": float(total_reward),
            "steps": steps,
            "traj": traj,
            "final_pos": final_pos,
            "goal_pos": goal_pos,
            "reached_goal": bool(reached)
        }


    def attach_agent(self, agent):
        self.agent = agent

    def reset_env(self):
        self.env.reset()
        start_pos = (0, self.env.tamanho[1] - 1)
        self.env.regista_agente(self.agent, start_pos)
        obs = self.env.observacaoPara(self.agent)
        self._last_obs = obs
        self._done = False
        self._steps = 0
        self._total_reward = 0.0
        return obs

    def step_once_for_visualiser(self, action=None):
        """
        Executa 1 tick do simulador (útil para UI ou debug).
        """
        if self._done:
            self.reset_env()

        obs = self._last_obs

        # escolher ação
        if action is None:
            action = self.agent.age(obs)

        # aplicar ação
        reward, done, info = self.env.agir(action, self.agent)

        # nova observação
        obs2 = self.env.observacaoPara(self.agent)

        # atualizar estado interne
        self._last_obs = obs2
        self._done = bool(done)
        self._steps += 1
        self._total_reward += reward

        if hasattr(self.env, "atualizacao"):
            self.env.atualizacao()

        return obs2, reward, done, info