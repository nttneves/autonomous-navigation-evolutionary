# algorithms/training_manager.py

import numpy as np
from model.model import create_mlp
from algorithms.genetic import set_weights_vector
from algorithms.trainer import EvolutionTrainer   # a classe que construímos antes

class EvolutionTrainingManager:
    """
    Classe que gere o processo de evolução, impressão de resultados
    e decisão do utilizador de guardar o modelo final.
    """

    def __init__(self,
                 env_factory,
                 input_dim: int = 10,
                 generations: int = 40,
                 population_size: int = 30,
                 max_steps: int = 200):

        self.env_factory = env_factory
        self.input_dim = input_dim
        self.generations = generations
        self.population_size = population_size
        self.max_steps = max_steps

        # Criar trainer genético robusto
        self.trainer = EvolutionTrainer(
            model_builder=lambda: create_mlp(self.input_dim),
            pop_size=self.population_size,
            archive_prob=0.1,
            elite_fraction=0.05
        )

    # -----------------------------------------------------------
    def run_training(self):
        """
        Treina durante N gerações e imprime resultados iguais à versão antiga.
        """
        print(f"\n🚀 INICIAR TREINO EVOLUTIVO")
        print(f"> Gerações: {self.generations}")
        print(f"> População: {self.population_size}")
        print(f"> Input dim: {self.input_dim}\n")

        history = self.trainer.train(
            env_factory=self.env_factory,
            max_steps=self.max_steps,
            generations=self.generations,
            episodes_per_individual=1,
            verbose=True
        )

        # Encontrar o melhor indivíduo da geração final
        best_agent = self.trainer.get_champion_agent()
        best_genome = best_agent.genoma

        # Resumo final igual ao antigo
        print("\n🏁 Evolução terminada!")
        print("Melhor genoma (primeiros 10 valores):")
        print(best_genome[:10])

        print("\n📈 Histórico (novelty média por geração):")
        print([round(h['mean_novelty'], 4) for h in history])

        print("\n📈 Histórico (novelty máxima por geração):")
        print([round(h['max_novelty'], 4) for h in history])

        # Diálogo final
        save = input("\n💾 Queres guardar o modelo final? (s/n): ").strip().lower()

        if save == "s":
            print("📦 A criar modelo final...")
            model = create_mlp(self.input_dim)
            set_weights_vector(model, best_genome)
            model.save("best_agent_model.keras")
            print("✅ Modelo guardado em best_agent_model.keras")

        print("\n👍 Treino concluído!\n")

        # devolver tudo para registro
        return {
            "history": history,
            "best_genome": best_genome,
            "best_agent": best_agent
        }