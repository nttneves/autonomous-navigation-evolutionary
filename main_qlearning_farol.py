from environments.environment_farol import FarolEnv
from algorithms.qlearning_trainer import FarolObservationDiscretizer
from agents.qlearning_agent import QLearningAgent
from simulator.simulator import Simulator

discretizer = FarolObservationDiscretizer()
agent = QLearningAgent.load(
    "model/best_agent_qlearning_farol.pkl",
    discretizer
)

env = FarolEnv((50,50), dificuldade=1, max_steps=300)
sim = Simulator(env, max_steps=300)
sim.agentes[agent.id] = agent

sim.reset_env()
sim.run_episode(render=True)