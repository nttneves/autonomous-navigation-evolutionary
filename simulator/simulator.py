# simulator.py
import json
from environments.environment_maze import MazeEnv
from environments.environment_farol import FarolEnv
from environments.environment import Enviroment
from agents.evolved_agent import EvolvedAgent
from agents.fixed_policy_agent import FixedPolicyAgent


class Simulator:
    def __init__(self, env: Enviroment, max_steps: int = 500):
        self.env = env
        self.max_steps = max_steps
        self.agentes = {}   # id → agente
        self.renderer = None

        # estado interno (visualiser)
        self._last_obs = None
        self._done = False
        self._steps = 0
        self._total_reward = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)

        env_name = data.get("ambiente")
        tamanho = tuple(data.get("tamanho", (21, 21)))
        dificuldade = data.get("dificuldade", 0)
        max_steps = data.get("max_steps", 200)

        if env_name == "farol":
            env = FarolEnv(tamanho=tamanho, dificuldade=dificuldade, max_steps=max_steps)
        elif env_name == "labirinto":
            env = MazeEnv(dificuldade=dificuldade, max_steps=max_steps)
        else:
            raise ValueError("Ambiente desconhecido")

        sim = cls(env, max_steps)

        for ficheiro in data.get("agentes", []):
            with open(ficheiro, "r") as fa:
                a_data = json.load(fa)

            tipo = a_data.get("tipo")
            if tipo == "fixed":
                ag = FixedPolicyAgent.cria(ficheiro)
            elif tipo == "evolved":
                ag = EvolvedAgent.cria(ficheiro)
            else:
                raise ValueError("Tipo de agente desconhecido.")

            sim.agentes[ag.id] = ag

        return sim

    # ------------------------------------------------------------------
    def listaAgentes(self):
        return list(self.agentes.values())

    # ------------------------------------------------------------------
    def run_episode(self, render=False):

        if render:
            from environments.renderer import EnvRenderer
            self.renderer = EnvRenderer(self.env)

        # reset ambiente
        self.env.reset()

        start_pos = (1, self.env.tamanho[1] - 1)

        for ag in self.agentes.values():
            self.env.regista_agente(ag, start_pos)
            ag.posicao = start_pos

        agent = self.listaAgentes()[0]

        traj = [start_pos]
        total_reward = 0.0
        steps = 0
        done = False

        obs = self.env.observacaoPara(agent)
        agent.observacao(obs)

        while not done and steps < self.max_steps:

            if render:
                alive = self.renderer.draw(self.env.posicoes_agentes)
                if not alive:
                    break

            action = agent.age()

            # --------------------------------------------------
            # MOVIMENTO (sem reward)
            # --------------------------------------------------
            prev_pos, new_pos, info = self.env.agir(action, agent)

            # --------------------------------------------------
            # REWARD (vem do ambiente!)
            # --------------------------------------------------
            reward, done = self.env.compute_reward(
                agent=agent,
                prev_pos=prev_pos,
                new_pos=new_pos,
                info=info
            )

            total_reward += reward
            agent.avaliacaoEstadoAtual(reward)

            obs = self.env.observacaoPara(agent)
            agent.observacao(obs)

            traj.append(new_pos)

            self.env.atualizacao()
            steps += 1

        if self.renderer:
            self.renderer.close()
            self.renderer = None

        final_pos = self.env.get_posicao_agente(agent)
        goal_pos = getattr(self.env, "goal_pos", None)
        reached = (goal_pos is not None and final_pos == goal_pos)

        return {
            "total_reward": float(total_reward),
            "steps": steps,
            "traj": traj,
            "final_pos": final_pos,
            "goal_pos": goal_pos,
            "reached_goal": reached
        }

    # ------------------------------------------------------------------
    def reset_env(self):
        agent = self.listaAgentes()[0]

        self.env.reset()
        start = (1, self.env.tamanho[1] - 1)
        self.env.regista_agente(agent, start)
        agent.posicao = start

        obs = self.env.observacaoPara(agent)
        agent.observacao(obs)

        self._last_obs = obs
        self._done = False
        self._steps = 0
        self._total_reward = 0.0

        return obs

    # ------------------------------------------------------------------
    def step_once_for_visualiser(self, action=None):
        agent = self.listaAgentes()[0]

        if self._done:
            self.reset_env()

        if action is None:
            action = agent.age()

        prev_pos, new_pos, info = self.env.agir(action, agent)

        reward, done = self.env.compute_reward(
            agent=agent,
            prev_pos=prev_pos,
            new_pos=new_pos,
            info=info
        )

        agent.avaliacaoEstadoAtual(reward)

        obs2 = self.env.observacaoPara(agent)
        agent.observacao(obs2)

        self._last_obs = obs2
        self._done = done
        self._steps += 1
        self._total_reward += reward

        self.env.atualizacao()

        return obs2, reward, done, info