# trainer.py
import numpy as np
import copy
from typing import Callable, List, Dict, Any
from agents.evolved_agent import EvolvedAgent
from algorithms.genetic import GeneticNoveltyTrainer, get_weights_vector, set_weights_vector
from environments.environment import Enviroment
import tensorflow as tf

class EvolutionTrainer:
    """
    Wrapper que usa GeneticNoveltyTrainer para evoluir vetores de pesos
    e avalia cada indivíduo no ambiente fornecido.
    """

    def __init__(self,
                 model_builder: Callable[[], Any],
                 pop_size: int = 50,
                 archive_prob: float = 0.1,
                 elite_fraction: float = 0.05,
                 seed: int = 42):
        np.random.seed(seed)

        self.model_builder = model_builder
        self.ga = GeneticNoveltyTrainer(model_builder, pop_size=pop_size,
                                        archive_prob=archive_prob,
                                        elite_fraction=elite_fraction)

        # track do melhor genoma global (e sua pontuação média)
        self.best_genome = None
        self.best_score = -np.inf

    def _evaluate_genome(self, genome: np.ndarray, env, max_steps: int):
        """
        Avalia UM genoma no ambiente fornecido.
        Retorna: total_reward (float), behaviour_char (np.ndarray)
        behaviour_char: [mean_x_norm, mean_y_norm, var_x_norm, var_y_norm]
        """

        # constrói modelo e agente com os pesos do genoma
        model = self.model_builder()
        set_weights_vector(model, genome)

        agent = EvolvedAgent(id="eval_agent", model=model)

        # Reset do ambiente
        env.reset()

        # posicionamento inicial: canto inferior esquerdo (0, h-1)
        start_pos = (0, env.tamanho[1] - 1)
        env.regista_agente(agent, start_pos)
        agent.posicao = start_pos

        total_reward = 0.0
        steps = 0
        done = False

        # coletar trajetória
        traj_positions = []
        traj_positions.append(start_pos)

        # primeira observação
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        while not done and steps < max_steps:
            action = agent.age()
            reward, done, info = env.agir(action, agent)
            agent.avaliacaoEstadoAtual(reward)
            total_reward += reward

            # nova observação
            obs = env.observacaoPara(agent)
            agent.observacao(obs)

            # regista posição atual
            pos = env.get_posicao_agente(agent)
            traj_positions.append(pos)

            # atualiza ambiente
            if hasattr(env, "atualizacao"):
                env.atualizacao()

            steps += 1

        final_pos = env.get_posicao_agente(agent)

        if final_pos is None:
            behaviour = np.array([0.0, 0.0], dtype=np.float32)
        else:
            fx, fy = final_pos
            w, h = env.tamanho
            behaviour = np.array([
                fx / float(w-1),
                fy / float(h-1)
            ], dtype=np.float32)

        return float(total_reward), behaviour

    def evaluate_population(self, env_factory, max_steps, episodes_per_individual=3):
        """
        Avalia toda a população corrente usando vários episódios por indivíduo.
        O fitness final = média das recompensas obtidas nos episódios.
        O behaviour (bc) final = média dos behaviours individuais.
        """

        pop = self.ga.population
        pop_size = len(pop)

        fitnesses = []
        behaviours = []

        for i in range(pop_size):
            genome = pop[i]

            total_reward = 0.0
            indiv_bcs = []

            for ep in range(episodes_per_individual):
                # criar ambiente novo
                env = env_factory()

                # avaliar 1 episódio
                reward, bc = self._evaluate_genome(genome, env, max_steps)

                total_reward += reward
                indiv_bcs.append(np.array(bc, dtype=np.float32))

            # fitness = média das recompensas
            final_fitness = total_reward / float(episodes_per_individual)

            # behaviour = média dos BCs
            bc_mean = np.mean(np.stack(indiv_bcs, axis=0), axis=0)

            fitnesses.append(float(final_fitness))
            behaviours.append(np.array(bc_mean, dtype=np.float32).copy())  # copiar sempre!

        return fitnesses, behaviours

    def evaluate_genome_multiple(self, genome: np.ndarray, env_factory: Callable[[], Any], max_steps: int, n_eval: int = 5):
        """
        Re-avalia um genoma em n_eval episódios diferentes (usando env_factory).
        Retorna a média das recompensas e uma média do BC.
        Útil para determinar se um genoma é robusto antes de o guardar.
        """
        total = 0.0
        bcs = []
        for _ in range(n_eval):
            env = env_factory()
            r, bc = self._evaluate_genome(genome, env, max_steps)
            total += r
            bcs.append(np.array(bc, dtype=np.float32))
        mean_reward = total / float(n_eval)
        mean_bc = np.mean(np.stack(bcs, axis=0), axis=0)
        return mean_reward, mean_bc

    def train(self,
            env_factory,
            max_steps: int,
            generations: int = 50,
            episodes_per_individual: int = 3,
            verbose: bool = True,
            champion_eval_episodes: int = 5):
        """
        Executa 'generations' ciclos de evolução genética.

        champion_eval_episodes: quando aparece um candidato melhor na geração,
                                re-avalia-o com 'champion_eval_episodes' episódios
                                antes de aceitar como novo best_global.
        """

        history = []

        for gen in range(1, generations + 1):

            # Avaliação da população completa (multi-episódios por indivíduo)
            fitnesses, behaviours = self.evaluate_population(
                env_factory,
                max_steps,
                episodes_per_individual
            )

            # Deep-copy dos behaviours (BCs)
            pop_bcs = [np.array(b, dtype=np.float32).copy() for b in behaviours]

            # Evolução da população (novelty search + GA)
            mean_novelty, max_novelty = self.ga.evolve(pop_bcs)

            # Estatísticas da geração
            mean_fitness = float(np.mean(fitnesses))
            max_fitness = float(np.max(fitnesses))
            best_idx = int(np.argmax(fitnesses))
            candidate_genome = self.ga.population[best_idx]

            # Re-avaliação robusta do candidato para evitar salvar "lucky" genomes
            cand_score, _ = self.evaluate_genome_multiple(candidate_genome, env_factory, max_steps, n_eval=champion_eval_episodes)

            if cand_score > self.best_score:
                self.best_score = cand_score
                self.best_genome = candidate_genome.copy()
                if verbose:
                    print(f"--> Novo BEST global (gen {gen}) score={cand_score:.4f}")

            # Guardar histórico
            history.append({
                "generation": gen,
                "mean_fitness": mean_fitness,
                "max_fitness": max_fitness,
                "mean_novelty": mean_novelty,
                "max_novelty": max_novelty,
                "best_idx": best_idx,
                "best_score_global": float(self.best_score) if self.best_score is not None else None
            })

            if verbose:
                print(
                    f"[Gen {gen}] "
                    f"fitness: mean={mean_fitness:.4f} max={max_fitness:.4f} | "
                    f"novelty: mean={mean_novelty:.4f} max={max_novelty:.4f} | "
                    f"best_idx={best_idx}"
                )

        return history

    def get_champion_agent(self):
        """
        Devolve um EvolvedAgent com o melhor genoma actual (best_genome).
        """
        if self.best_genome is None:
            # fallback: devolve um agente com o primeiro genoma da população
            pop = self.ga.population
            best = pop[0]
            model = self.model_builder()
            set_weights_vector(model, best)
            agent = EvolvedAgent(id="champion", model=model)
            agent.set_genoma(best)
            return agent

        model = self.model_builder()
        set_weights_vector(model, self.best_genome)
        agent = EvolvedAgent(id="champion", model=model)
        agent.set_genoma(self.best_genome)
        return agent

    def save_champion(self, path: str, env_factory: Callable[[], Any], max_steps: int, n_eval: int = 10):
        """
        Salva o modelo do champion em disco. Re-avalia o champion em n_eval episódios
        e só salva se a média for razoável (proteção extra contra casos degenerados).
        """
        if self.best_genome is None:
            raise RuntimeError("Não existe best_genome. Executa train() primeiro.")

        score, _ = self.evaluate_genome_multiple(self.best_genome, env_factory, max_steps, n_eval=n_eval)
        model = self.model_builder()
        set_weights_vector(model, self.best_genome)

        # podes ajustar o threshold se quiseres
        model.save(path)
        return score