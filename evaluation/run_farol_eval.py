import numpy as np

from evaluation.evaluator import AgentEvaluator
from environments.environment_farol import FarolEnv

from agents.fixed_policy_agent import FixedPolicyAgent
from agents.evolved_agent import EvolvedAgent
from agents.qlearning_agent import QLearningAgent

from main_qlearning_farol import test_agent_same_protocol
from model.model import create_mlp
from algorithms.genetic import set_weights_vector


def farol_env_factory():
    return FarolEnv(
        tamanho=(50, 50),
        dificuldade=2,
        max_steps=1000,
        seed=None
    )


# ================= FIXED =================
fixed = AgentEvaluator(
    FixedPolicyAgent(id="fixed"),
    farol_env_factory,
    test_agent_same_protocol,
    name="Fixed"
)

# ================= EVOLVED =================
genome = np.load("model/best_agent_farol.npy")
model = create_mlp(input_dim=12)
set_weights_vector(model, genome)

evolved = AgentEvaluator(
    EvolvedAgent(id="evolved", model=model),
    farol_env_factory,
    test_agent_same_protocol,
    name="Evolved"
)

# ================= Q-LEARNING =================
qlearning = AgentEvaluator(
    QLearningAgent.load("model/best_agent_qlearning_farol.pkl"),
    farol_env_factory,
    test_agent_same_protocol,
    name="Q-learning"
)

evaluators = [fixed, evolved, qlearning]

for e in evaluators:
    e.run()
    e.save_json(f"results/farol/test/{e.name.lower()}.json")

AgentEvaluator.plot_steps_comparison(
    evaluators,
    "results/farol/test/steps_comparison.png",
    "Farol — Passos médios até ao objetivo"
)

AgentEvaluator.plot_success_rate(
    evaluators,
    "results/farol/test/success_rate.png",
    "Farol — Taxa de Sucesso"
)