from environments.environment_maze import MazeEnv
from environments.environment_farol import FarolEnv
from agents.fixed_policy_agent import FixedPolicyAgent
from simulator.simulator import Simulator

#env = FarolEnv(tamanho=(30,30), dificuldade=3, max_steps=300)
# env = MazeEnv(dificuldade=2, max_steps=600)
# agent = FixedPolicyAgent(id="rnd")
# sim = Simulator(env, max_steps=600)
# sim.agentes = {"rnd": agent}

sim = Simulator.cria("files/simulator_maze.json")

#sim = Simulator.cria("files/simulator_farol.json")

result = sim.run_episode(render=True)
print(result)