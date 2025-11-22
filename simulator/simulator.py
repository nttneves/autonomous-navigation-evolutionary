# simulator.py
import json
from environments.environment_farol import FarolEnv
from agents.evolved_agent import EvolvedAgent
from agents.random_agent import RandomAgent

class Simulator:

  # TODO: PARA LER DO FICHEIRO O UNICO PARAMETRO QUE PRECISO É O FICHEIRO
    def __init__(self, env, max_steps: int):
        self.env = env
        self.max_steps = max_steps
        self.agentes = {}   # id → agente
        self.renderer = None

    # ------------------------------------------------------------------
    # TODO: O FICHEIRO DEVE TER O AMBIENTE, TAMANHO, DIFICULDADE, MAX_STEPS, NUMERO DE AGENTES E CAMINHO PARA O FICHEIRO DE CRIAR AGENTES
    @classmethod
    def cria(cls, ficheiro_json: str):
        """
        Lê a configuração e constrói o simulador.
        Espera um JSON com:
        {
            "ambiente": "farol",
            "tamanho": [21,21],
            "dificuldade": 0,
            "max_steps": 200,
            "agentes": ["ficheiro1.json", "ficheiro2.json"]
        }
        """
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

            if tipo == "random":
                ag = RandomAgent.cria(ficheiro)
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
            from environments.renderer_farol import FarolRenderer
            self.renderer = FarolRenderer(self.env)

        self.env.reset()

        # registar agentes
        start_x = 0
        start_y = self.env.tamanho[1] - 1

        for ag in self.listaAgentes():
            self.env.regista_agente(ag, (start_x, start_y))

        agent = self.listaAgentes()[0]

        traj = [self.env.get_posicao_agente(agent)]
        total_reward = 0.0
        steps = 0
        done = False

        # primeira observação
        obs = self.env.observacaoPara(agent)
        agent.observacao(obs)

        # loop
        while not done and steps < self.max_steps:

            # renderização
            if render:
                alive = self.renderer.draw(self.env.posicoes_agentes)
                if not alive:
                    break  # janela fechada pelo utilizador

            accao = agent.age()
            reward, done, info = self.env.agir(accao, agent)
            agent.avaliacaoEstadoAtual(reward)

            obs = self.env.observacaoPara(agent)
            agent.observacao(obs)

            pos = self.env.get_posicao_agente(agent)
            traj.append(pos)

            self.env.atualizacao()
            steps += 1

        if self.renderer:
            self.renderer.close()
            self.renderer = None

        final_pos = self.env.get_posicao_agente(agent)
        goal_pos = getattr(self.env, "farol_pos", None)
        reached = (final_pos == goal_pos)

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
        """Útil para visualizadores ou step-by-step."""
        agent = self.listaAgentes()[0]

        self.env.reset()
        start = (0, self.env.tamanho[1] - 1)
        self.env.regista_agente(agent, start)

        obs = self.env.observacaoPara(agent)
        agent.observacao(obs)

        self._last_obs = obs
        self._done = False
        self._steps = 0
        self._total_reward = 0.0

        return obs

    # ------------------------------------------------------------------
    def step_once_for_visualiser(self, action=None):
        """Executa 1 tick útil para UIs."""
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