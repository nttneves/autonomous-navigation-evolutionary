# simulator.py
import json
from environments.enviromnent_maze import MazeEnv
from environments.environment import Enviroment
from environments.environment_farol import FarolEnv
from agents.evolved_agent import EvolvedAgent
from agents.fixed_policy_agent import FixedPolicyAgent

class Simulator:
  # TODO: PARA LER DO FICHEIRO O UNICO PARAMETRO QUE PRECISO É O FICHEIRO
    def __init__(self, env: Enviroment, max_steps: int):
        self.env = env
        self.max_steps = max_steps
        self.agentes = {}   # id → agente
        self.renderer = None

    # ------------------------------------------------------------------
    # TODO: O FICHEIRO DEVE TER O AMBIENTE, TAMANHO, DIFICULDADE, MAX_STEPS, NUMERO DE AGENTES E CAMINHO PARA O FICHEIRO DE CRIAR AGENTES
    @classmethod
    def cria(cls, ficheiro_json: str):
        with open(ficheiro_json, "r") as f:
            data = json.load(f)

        # construir ambiente
        if data["ambiente"] == "farol":
            env = FarolEnv(
                tamanho=tuple(data["tamanho"]),
                dificuldade=data.get("dificuldade", 0),
                max_steps=data.get("max_steps", 200)
            )
        else:
            raise ValueError("Ambiente desconhecido")

        sim = cls(env, data.get("max_steps", 200))

        # carregar agentes
        for ficheiro in data["agentes"]:
            with open(ficheiro, "r") as fa:
                a_data = json.load(fa)

            tipo = a_data["tipo"]

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

        # iniciar renderer se necessário
        if render:
            from environments.renderer import EnvRenderer
            self.renderer = EnvRenderer(self.env)

        # reset do ambiente
        self.env.reset()

        # posição inicial
        start_pos = (0, self.env.tamanho[1] - 1)

        # registar agentes corretamente
        for ag in self.agentes.values():
            self.env.regista_agente(ag, start_pos)
            ag.posicao = start_pos  # CRÍTICO: manter consistência interna

        agent = self.listaAgentes()[0]

        traj = [self.env.get_posicao_agente(agent)]
        total_reward = 0.0
        steps = 0
        done = False

        # primeira observação
        obs = self.env.observacaoPara(agent)
        agent.observacao(obs)

        # loop principal
        while not done and steps < self.max_steps:

            # desenhar no pygame
            if render:
                alive = self.renderer.draw(self.env.posicoes_agentes)
                if not alive:
                    break

            # agente decide
            accao = agent.age()

            # ambiente reage
            reward, done, info = self.env.agir(accao, agent)
            total_reward += reward
            agent.avaliacaoEstadoAtual(reward)

            # nova observação
            obs = self.env.observacaoPara(agent)
            agent.observacao(obs)

            # guardar trajetória
            pos = self.env.get_posicao_agente(agent)
            agent.posicao = pos   # CRÍTICO: atualizar posição interna
            traj.append(pos)

            # atualizar ambiente
            self.env.atualizacao()

            steps += 1

        # fechar janela se aberta
        if self.renderer:
            self.renderer.close()
            self.renderer = None

        final_pos = self.env.get_posicao_agente(agent)
        # procurar atributo que representa a meta
        goal_pos = None
        if hasattr(self.env, "farol_pos"):
            goal_pos = self.env.farol_pos
        elif hasattr(self.env, "saida_pos"):
            goal_pos = self.env.saida_pos

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
        start = (0, self.env.tamanho[1] - 1)
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

        obs = self._last_obs

        if action is None:
            action = agent.age()

        reward, done, info = self.env.agir(action, agent)
        agent.avaliacaoEstadoAtual(reward)

        obs2 = self.env.observacaoPara(agent)
        agent.observacao(obs2)

        self._last_obs = obs2
        self._done = done
        self._steps += 1
        self._total_reward += reward

        self.env.atualizacao()

        return obs2, reward, done, info