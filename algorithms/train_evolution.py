# algorithms/train_evolution.py
import numpy as np
import random
from simulator.simulator import Simulator
from agents.evolved_agent import EvolvedAgent
from algorithms.genetic import GeneticNoveltyTrainer, novelty_score, set_weights_vector
from algorithms.evaluate import evaluate_individual
from model.model import create_mlp
from environments.environment_farol import FarolEnv

# -----------------------------------------------------
# # AINDA NÃO FOI ALTERADO - VERSÃO ANTIGA
# -----------------------------------------------------


def train_evolution(
    generations=50,
    population_size=40,
    input_dim: int = None,
    tamanho=(21, 21),
    dificuldade=0,
    max_steps=200,
    seed: int = 42,
):

    random.seed(seed)
    np.random.seed(seed)

    print("🔧 A iniciar ambiente e simulador...")

    env = FarolEnv(
        tamanho=tamanho,
        dificuldade=dificuldade,
        max_steps=max_steps
    )

    simulator = Simulator(env, max_steps=max_steps)

    # ==========================================================
    # 1. INFERIR input_dim AUTOMATICAMENTE
    # ==========================================================
    inferred_input_dim = None

    try:
        dummy = EvolvedAgent(id="__dummy__", dim_input_rn=1)
        _instalar_sensores(dummy)

        start_pos = (0, env.tamanho[1] - 1)
        env.reset()
        env.regista_agente(dummy, start_pos)

        obs = env.observacaoPara(dummy) or {}
        inferred_input_dim = sum(np.asarray(v).flatten().shape[0] for v in obs.values())

    except Exception as e:
        print("⚠️ Falha ao inferir input_dim automaticamente:", e)

    if inferred_input_dim is None:
        if input_dim is None:
            raise ValueError("Não foi possível inferir input_dim. Tens de o fornecer manualmente.")
        final_input_dim = int(input_dim)
        print(f"⚠️ A usar input_dim fornecido: {final_input_dim}")
    else:
        final_input_dim = int(inferred_input_dim)
        print(f"✅ input_dim inferido automaticamente: {final_input_dim}")

    # ==========================================================
    # 2. CRIAR TREINADOR GENÉTICO
    # ==========================================================
    trainer = GeneticNoveltyTrainer(
        model_builder=lambda: create_mlp(final_input_dim),
        pop_size=population_size,
        archive_prob=0.10
    )

    history_mean = []
    history_max = []

    best_global_novelty = -np.inf
    best_global_genome = None
    best_global_bc = None
    best_global_generation = None

    print(f"🚀 Evolução iniciada: {population_size} indivíduos × {generations} gerações.\n")

    # ==========================================================
    # 3. LOOP DE GERAÇÕES
    # ==========================================================
    for gen in range(generations):

        behaviours = []

        for idx, genome in enumerate(trainer.population):

            # Avaliar indivíduo
            bc = evaluate_individual(
                weights=genome,
                create_model=lambda: create_mlp(final_input_dim),
                simulator=simulator
            )
            behaviours.append(np.asarray(bc, dtype=np.float32))

        # Calcular novelty
        nov_scores = []
        for b in behaviours:
            ns = novelty_score(b, behaviours, trainer.archive, k=10)
            nov_scores.append(ns)

            if ns > best_global_novelty:
                best_global_novelty = ns
                best_global_genome = trainer.population[len(nov_scores)-1]
                best_global_bc = b.copy()
                best_global_generation = gen

        mean_nov = float(np.mean(nov_scores))
        max_nov = float(np.max(nov_scores))

        history_mean.append(mean_nov)
        history_max.append(max_nov)

        trainer.evolve(behaviours)

        print(
            f"Geração {gen+1}/{generations} | "
            f"Novelty média={mean_nov:.4f} | Máx={max_nov:.4f} | "
            f"BestGlobal={best_global_novelty:.4f} (gen {best_global_generation})"
        )

    print("\n🏁 Evolução terminada!")

    # fallback
    if best_global_genome is None:
        best_global_genome = trainer.population[0]
        best_global_bc = behaviours[0]

    # ==========================================================
    # 4. MOSTRAR RESUMO FINAL (igual ao main original)
    # ==========================================================

    print("\nRESULTADO FINAL DO TESTE:")
    print("Melhor genoma (primeiros 10 valores):", best_global_genome[:10])
    print("Histórico novelty média:", history_mean)
    print("Histórico novelty máxima:", history_max)

    # ==========================================================
    # 5. Perguntar se quer guardar o modelo
    # ==========================================================

    save = input("\n💾 Queres guardar o modelo final? (s/n): ").strip().lower()

    if save == "s":
        print("📦 A criar modelo final...")
        model = create_mlp(final_input_dim)
        set_weights_vector(model, best_global_genome)
        model.save("best_agent_model.keras")
        print("✅ Modelo guardado em best_agent_model.keras")

    return {
        "melhor_genoma": best_global_genome,
        "melhor_bc": best_global_bc,
        "melhor_novelty": float(best_global_novelty),
        "melhor_geracao": best_global_generation,
        "history_mean_nov": history_mean,
        "history_max_nov": history_max,
        "input_dim": final_input_dim
    }


# ==========================================================
# INSTALAR SENSORES NO AGENTE
# ==========================================================
def _instalar_sensores(agent):
    """Instala os sensores padrão num agente."""
    from sensor.sensor_objeto import sensor_objeto
    from sensor.sensor_obstaculo import sensor_obstaculo

    s_obj = sensor_objeto(alcance=10)
    s_obj.dim = 8

    s_obs = sensor_obstaculo(alcance=10)
    s_obs.dim = 4

    agent.instala("sensor_objeto", s_obj)
    agent.instala("sensor_obstaculo", s_obs)