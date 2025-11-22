# main.py
from environments.environment_farol import FarolEnv
from agents.random_agent import RandomAgent
from simulator.simulator import Simulator

def main():
    env = FarolEnv(
        tamanho=(21,21),
        dificuldade=0,
        max_steps=200
    )

    agent = RandomAgent(id="A1", num_acoes=4)

    sim = Simulator(env, max_steps=200)
    sim.agentes[agent.id] = agent

    resultado = sim.run_episode(render=True)

    print("==== RESULTADO ====")
    print(resultado)

if __name__ == "__main__":
    main()