import numpy as np
import os

from evaluation.evaluator import AgentEvaluator, test_agent_same_protocol
from environments.environment_maze import MazeEnv

from agents.fixed_policy_agent import FixedPolicyAgent
from agents.evolved_agent import EvolvedAgent
from agents.qlearning_agent import QLearningAgent

from algorithms.qlearning_trainer import MazeObservationDiscretizer

from model.model import create_mlp
from algorithms.genetic import set_weights_vector


MAX_STEPS = 1000
DIFFICULTIES = [0, 1, 2]


# ============================================================
# Loop por dificuldade
# ============================================================

for DIFFICULTY in DIFFICULTIES:

    print(f"\n===== AVALIAÇÃO MAZE — DIFICULDADE {DIFFICULTY} =====")

    def maze_env_factory():
        return MazeEnv(
            dificuldade=DIFFICULTY,
            max_steps=MAX_STEPS
        )

    # ================= FIXED =================
    fixed = AgentEvaluator(
        FixedPolicyAgent(id=f"fixed_d{DIFFICULTY}"),
        maze_env_factory,
        test_agent_same_protocol,
        n_runs=30,
        max_steps=MAX_STEPS,
        name="Fixed"
    )

    # ================= EVOLVED =================
    genome = np.load("model/best_agent_maze.npy")
    
    model = create_mlp(input_dim=12)
    set_weights_vector(model, genome)

    evolved = AgentEvaluator(
        EvolvedAgent(id=f"evolved_d{DIFFICULTY}", model=model),
        maze_env_factory,
        test_agent_same_protocol,
        n_runs=30,
        max_steps=MAX_STEPS,
        name="Evolved"
    )

    # ================= Q-LEARNING =================
    discretizer = MazeObservationDiscretizer()

    qlearning = AgentEvaluator(
        QLearningAgent.load(
            "model/best_agent_qlearning_maze.pkl",
            discretizer=discretizer,
            id=f"qlearning_d{DIFFICULTY}"
        ),
        maze_env_factory,
        test_agent_same_protocol,
        n_runs=30,
        max_steps=MAX_STEPS,
        name="Q-learning"
    )

    evaluators = [fixed, evolved, qlearning]

    # ========================================================
    # Executar avaliações
    # ========================================================
    out_dir = f"results/maze/test/difficulty_{DIFFICULTY}"
    os.makedirs(out_dir, exist_ok=True)

    for e in evaluators:
        e.run()
        e.save_json(f"{out_dir}/{e.name.lower()}.json")

    # ========================================================
    # Gráficos comparativos
    # ========================================================
    AgentEvaluator.plot_steps_comparison(
        evaluators,
        f"{out_dir}/steps_comparison.png",
        f"Maze (Dificuldade {DIFFICULTY}) — Passos médios até ao objetivo"
    )

    AgentEvaluator.plot_success_rate(
        evaluators,
        f"{out_dir}/success_rate.png",
        f"Maze (Dificuldade {DIFFICULTY}) — Taxa de Sucesso"
    )