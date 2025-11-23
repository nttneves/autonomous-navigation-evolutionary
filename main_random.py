from environments.environment_farol import FarolEnv
from agents.random_agent import RandomAgent
from simulator.simulator import Simulator

env = FarolEnv()
agent = RandomAgent(id="rnd", num_acoes=4)
sim = Simulator(env, max_steps=200)
sim.agentes = {"rnd": agent}

result = sim.run_episode(render=True)
print(result)