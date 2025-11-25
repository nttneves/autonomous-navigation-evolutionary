from environments.enviromnent_maze import MazeEnv
from environments.environment_farol import FarolEnv
from agents.fixed_policy_agent import FixedPolicyAgent
from simulator.simulator import Simulator

env = FarolEnv()
agent = FixedPolicyAgent(id="rnd")
sim = Simulator(env, max_steps=200)
sim.agentes = {"rnd": agent}

result = sim.run_episode(render=True)
print(result)