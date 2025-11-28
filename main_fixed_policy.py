from environments.environment_maze import MazeEnv
from environments.environment_farol import FarolEnv
from agents.fixed_policy_agent import FixedPolicyAgent
from simulator.simulator import Simulator

# env = MazeEnv(dificuldade=2)
# agent = FixedPolicyAgent(id="rnd")
# sim = Simulator(env, max_steps=700)
# sim.agentes = {"rnd": agent}

sim = Simulator.cria("files/simulator_maze.json")

result = sim.run_episode(render=True)
print(result)