# algorithms/trainer.py
import numpy as np
from typing import Callable, Any, List
from agents.evolved_agent import EvolvedAgent
from algorithms.genetic import GeneticNoveltyTrainer, set_weights_vector
from environments.environment import Enviroment


class EvolutionTrainer:
    def __init__(
        self,
        model_builder: Callable[[], Any],
        pop_size: int = 100,
        archive_prob: float = 0.1,
        elite_fraction: float = 0.05,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.model_builder = model_builder
        self.ga = GeneticNoveltyTrainer(
            model_builder,
            pop_size=pop_size,
            archive_prob=archive_prob,
            elite_fraction=elite_fraction,
            seed=seed
        )
        self.best_genome = None
        self.best_score = -np.inf

    # =========================================================
    # Avaliação de UM genoma num ambiente
    # =========================================================
    def _evaluate_genome(self, genome, env: Enviroment, max_steps):
        model = self.model_builder()
        set_weights_vector(model, genome)
        agent = EvolvedAgent(id="eval", model=model)

        env.reset()
        start_pos = (0, env.tamanho[1] - 1)
        env.regista_agente(agent, start_pos)
        agent.posicao = start_pos

        total_reward = 0.0
        steps = 0
        done = False

        # observação inicial
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        # distância ao goal para shaping
        bx, by = env.goal_pos

        def dist(pos):
            dx = bx - pos[0]
            dy = by - pos[1]
            return np.sqrt(dx*dx + dy*dy)

        prev_pos = start_pos
        prev_dist = dist(prev_pos)

        # loop principal
        while not done and steps < max_steps:
            action = agent.age()
            reward, done, info = env.agir(action, agent)

            # reward shaping de aproximação
            pos_now = env.get_posicao_agente(agent)
            if pos_now is None:
                pos_now = prev_pos
            new_dist = dist(pos_now)

            reward += (prev_dist - new_dist) * 0.5
            prev_dist = new_dist

            agent.avaliacaoEstadoAtual(reward)
            total_reward += reward

            obs = env.observacaoPara(agent)
            agent.observacao(obs)
            prev_pos = pos_now

            if hasattr(env, "atualizacao"):
                env.atualizacao()

            steps += 1

        # BC = posição final + distância normalizada
        final_pos = env.get_posicao_agente(agent)
        w, h = env.tamanho
        maxd = np.sqrt((w-1)**2 + (h-1)**2)

        if final_pos is None:
            behaviour = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            fx, fy = final_pos
            d = np.sqrt((fx - bx)**2 + (fy - by)**2)
            behaviour = np.array([fx/(w-1), fy/(h-1), d/maxd], dtype=np.float32)

        return float(total_reward), behaviour

    # =========================================================
    # Avaliação normal (um ambiente)
    # =========================================================
    def evaluate_population(self, env_factory, max_steps, episodes_per_individual=3):
        pop = self.ga.population
        fitnesses = []
        behaviours = []

        for genome in pop:
            total = 0.0
            bcs = []
            for _ in range(episodes_per_individual):
                env = env_factory()
                r, bc = self._evaluate_genome(genome, env, max_steps)
                total += r
                bcs.append(bc)
            fitnesses.append(total / episodes_per_individual)
            behaviours.append(np.mean(np.stack(bcs), axis=0).astype(np.float32))

        return fitnesses, behaviours

    # =========================================================
    # Avaliação multi-ambiente (Curriculum Learning)
    # =========================================================
    def evaluate_population_multi(
        self,
        env_factories: List[Callable[[], Enviroment]],
        max_steps: int,
        episodes_per_individual: int = 3
    ):
        pop = self.ga.population
        fitnesses = []
        behaviours = []

        for genome in pop:
            all_rewards = []
            all_bcs = []

            # a cada geração cada indivíduo é testado em TODOS os ambientes
            for make_env in env_factories:
                for _ in range(episodes_per_individual):
                    env = make_env()
                    r, bc = self._evaluate_genome(genome, env, max_steps)
                    all_rewards.append(r)
                    all_bcs.append(bc)

            fitnesses.append(np.mean(all_rewards))
            behaviours.append(np.mean(np.stack(all_bcs), axis=0).astype(np.float32))

        return fitnesses, behaviours

    # =========================================================
    # Reavaliação mais robusta
    # =========================================================
    def evaluate_genome_multiple(self, genome, env_factory, max_steps, n_eval=8):
        total = 0.0
        bcs = []
        for _ in range(n_eval):
            env = env_factory()
            r, bc = self._evaluate_genome(genome, env, max_steps)
            total += r
            bcs.append(bc)
        return total / n_eval, np.mean(np.stack(bcs), axis=0)

    # =========================================================
    # TREINO: suportar 1 ambiente OU lista de ambientes
    # =========================================================
    def train(
            self,
            env_factories,
            max_steps: int,
            generations: int = 50,
            episodes_per_individual: int = 3,
            alpha: float = 0.7,
            verbose: bool = True,
            champion_eval_episodes: int = 8,
            external_generation_offset: int = 0
        ):
        
        history = []

        for gen in range(1, generations + 1):
            # --- avaliação normal ---
            fitnesses, behaviours = self.evaluate_population(env_factories, max_steps, episodes_per_individual)
            pop_bcs = [b.copy() for b in behaviours]

            mean_f = float(np.mean(fitnesses))
            max_f = float(np.max(fitnesses))
            best_idx = int(np.argmax(fitnesses))
            best_genome_gen = self.ga.population[best_idx].copy()

            cand_score, _ = self.evaluate_genome_multiple(
                best_genome_gen, env_factories, max_steps, n_eval=champion_eval_episodes
            )

            if cand_score > self.best_score:
                self.best_score = cand_score
                self.best_genome = best_genome_gen.copy()
                if verbose:
                    print(f"--> Novo BEST global (gen {external_generation_offset + gen}) score={cand_score:.4f}")

            # evolve
            mean_nov, max_nov = self.ga.evolve(pop_bcs, fitnesses=fitnesses, alpha=alpha)

            # --- geração global corrigida ---
            history.append({
                "generation": external_generation_offset + gen,
                "mean_fitness": mean_f,
                "max_fitness": max_f,
                "mean_novelty": mean_nov,
                "max_novelty": max_nov,
                "best_idx": best_idx,
                "best_score_global": float(self.best_score)
            })

            if verbose:
                print(f"[Gen {external_generation_offset + gen}] fitness: mean={mean_f:.4f} max={max_f:.4f} | "
                    f"novelty: mean={mean_nov:.4f} max={max_nov:.4f} | best_idx={best_idx}")

        return history

    # =========================================================
    def get_champion_agent(self):
        if self.best_genome is None:
            g = self.ga.population[0]
            model = self.model_builder()
            set_weights_vector(model, g)
            a = EvolvedAgent(id="champion", model=model)
            a.set_genoma(g)
            return a

        model = self.model_builder()
        set_weights_vector(model, self.best_genome)
        a = EvolvedAgent(id="champion", model=model)
        a.set_genoma(self.best_genome)
        return a

    # =========================================================
    def save_champion(self, path, env_factory, max_steps, n_eval=10, threshold=None):
        if self.best_genome is None:
            raise RuntimeError("No best genome")

        score, _ = self.evaluate_genome_multiple(
            self.best_genome, env_factory, max_steps, n_eval
        )

        if threshold is not None and score < threshold:
            return False, score

        model = self.model_builder()
        set_weights_vector(model, self.best_genome)
        model.save(path)
        return True, score