import numpy as np
import json
import os
import matplotlib.pyplot as plt


class AgentEvaluator:
    """
    Avaliação padronizada de agentes em modo de teste.
    Executa N episódios e calcula métricas agregadas.
    """

    def __init__(
        self,
        agent,
        env_factory,
        test_fn,
        n_runs: int = 30,
        max_steps: int = 1000,
        name: str = "agent"
    ):
        self.agent = agent
        self.env_factory = env_factory
        self.test_fn = test_fn
        self.n_runs = n_runs
        self.max_steps = max_steps
        self.name = name

        self.results = []

    # =========================================================
    # EXECUTAR TESTES
    # =========================================================
    def run(self):
        self.results.clear()

        for _ in range(self.n_runs):
            env = self.env_factory()

            if hasattr(self.agent, "reset"):
                self.agent.reset()

            if hasattr(self.agent, "epsilon"):
                self.agent.epsilon = 0.0  # teste puro

            res = self.test_fn(
                self.agent,
                env,
                max_steps=self.max_steps,
                render=False
            )

            self.results.append(res)

        return self.results

    # =========================================================
    # MÉTRICAS
    # =========================================================
    def summary(self):
        successes = [r["reached_goal"] for r in self.results]
        steps_success = [r["steps"] for r in self.results if r["reached_goal"]]

        return {
            "agent": self.name,
            "n_runs": self.n_runs,
            "success_rate": 100.0 * float(np.mean(successes)),
            "mean_steps_success": float(np.mean(steps_success)) if steps_success else None,
            "std_steps_success": float(np.std(steps_success)) if steps_success else None
        }

    # =========================================================
    # GUARDAR JSON
    # =========================================================
    def save_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "summary": self.summary(),
                    "raw_results": self.results
                },
                f,
                indent=4
            )

    # =========================================================
    # GRÁFICO COMPARATIVO — PASSOS MÉDIOS
    # =========================================================
    @staticmethod
    def plot_steps_comparison(evaluators, path, title):
        names = []
        means = []
        stds = []

        for ev in evaluators:
            s = ev.summary()
            if s["mean_steps_success"] is None:
                continue
            names.append(s["agent"])
            means.append(s["mean_steps_success"])
            stds.append(s["std_steps_success"])

        plt.figure(figsize=(7, 5))
        plt.bar(
            names,
            means,
            yerr=stds,
            capsize=6,
            edgecolor="black",
            linewidth=1.5
        )

        plt.ylabel("Passos médios até ao objetivo")
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()

    # =========================================================
    # GRÁFICO COMPARATIVO — TAXA DE SUCESSO
    # =========================================================
    @staticmethod
    def plot_success_rate(evaluators, path, title):
        names = [e.name for e in evaluators]
        rates = [e.summary()["success_rate"] for e in evaluators]

        plt.figure(figsize=(7, 5))
        plt.bar(
            names,
            rates,
            edgecolor="black",
            linewidth=1.5
        )

        plt.ylabel("Taxa de Sucesso (%)")
        plt.ylim(0, 100)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()