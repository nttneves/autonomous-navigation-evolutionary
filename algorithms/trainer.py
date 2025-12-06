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
        self._model_cache = {}   # cache de modelos indexados por genome_id

    # =========================================================
    # Cache + criação de modelo
    # =========================================================
    def _get_or_build_model(self, genome_id, genome):
        """
        Cria ou reutiliza um modelo SimpleMLP_RNN e aplica o genoma.
        Usa set_weights_vector (do genetic.py), não set_weights().
        """
        model = self._model_cache.get(genome_id)

        if model is None:
            model = self.model_builder()
            self._model_cache[genome_id] = model

        # aplica pesos ao modelo (genoma é vetor 1D)
        set_weights_vector(model, genome)
        return model

    # =========================================================
    # Avaliação de 1 genoma (1 episódio)
    # =========================================================
    def _evaluate_genome(self, genome, env: Enviroment, max_steps, genome_id=0):
        """
        Avalia um único genoma num único episódio no env passado.
        Retorna: (total_reward, behaviour_vector)
        """
        model = self._get_or_build_model(genome_id, genome)
        agent = EvolvedAgent(id="eval", model=model, sensores=True)

        # Reset do estado da RNN no agente (importante)
        if hasattr(agent, "reset"):
            agent.reset()

        env.reset()

        h = env.tamanho[1]
        start_pos = (1,h-1)
        #start_pos = (h-1,1)
        env.regista_agente(agent, start_pos)
        agent.posicao = start_pos

        total_reward = 0.0
        steps = 0
        done = False

        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        bx, by = env.goal_pos
        px, py = start_pos

        hypot = np.hypot
        dist = hypot(px - bx, py - by)
        prev_dist = dist
        traj_len = 0
        trajectory = [] # Novo

        is_maze = "maze" in env.__class__.__name__.lower()
        visited = {start_pos}

        while steps < max_steps and not done:
            action = agent.age()
            reward, done, info = env.agir(action, agent)

            pos_now = env.get_posicao_agente(agent) or (px, py)
            x, y = pos_now

            # reward shaping específico para maze
            if is_maze:
                # incentivo a mover-se (pequeno)
                reward += 1.0

                if info.get("collision", False):
                    # colisões penalizam
                    reward -= 8
                else:
                   
                   
                    # só recompensa exploração se for novo cell
                    if pos_now not in visited:
                        new_dist = hypot(x - bx, y - by)
                        delta = max(0, prev_dist - new_dist)
                        reward += delta * 10.0
                        prev_dist = new_dist
                        reward += 5.0  # bónus por novo cell
                        visited.add(pos_now)
                        #print(f"Novo cell visitado: {pos_now} | Distância ao objetivo: {new_dist:.2f}, Recompensa extra! {delta:.2f}")
                    else:
                        reward -= 4.5

                if pos_now == (bx, by) or info.get("reached_beacon", False):
                    reward += 500.0
                    print(f"Objetivo alcançado em {steps} passos!")
            else:
                # para outros ambientes: distância como shaping
                new_dist = hypot(x - bx, y - by)
                reward += (prev_dist - new_dist) * 5.0
                prev_dist = new_dist

        

            total_reward += reward
            agent.avaliacaoEstadoAtual(reward)

            traj_len += 1
            # observa novo estado
            agent.observacao(env.observacaoPara(agent))
            if steps % 5 == 0: # A cada 5 passos
                trajectory.append(pos_now)
            px, py = x, y

            if hasattr(env, "atualizacao"):
                env.atualizacao()

            steps += 1

        # Behaviour Characteristic final
        final_pos = env.get_posicao_agente(agent) or start_pos
        w, h = env.tamanho
        fx, fy = final_pos
        maxd = hypot(w - 1, h - 1) if (w > 1 and h > 1) else 1.0
        d_norm = dist / maxd

        unique_cells_norm = len(visited) / max(1, w * h)
        traj_len_norm = traj_len / max(1, max_steps)

        # Novo: Amostrar 5 posições (ou outro número fixo) da trajetória normalizada
        num_samples = 5
        indices = np.linspace(0, len(trajectory) - 1, num_samples, dtype=int)
        sampled_trajectory = [trajectory[i] for i in indices]
        # Normalizar as coordenadas X e Y
        w, h = env.tamanho
        traj_bcs = np.array([
            (x / max(1, w - 1), y / max(1, h - 1)) for x, y in sampled_trajectory
        ]).flatten()
        # Behaviour Characteristic final
        behaviour = np.concatenate([
            traj_bcs, # As 5 posições (10 valores)
            np.array([
                fx / max(1, (w - 1)),
                fy / max(1, (h - 1)),
                d_norm,
                unique_cells_norm,
                traj_len_norm
            ], dtype=np.float32)
        ])

        # return raw total_reward (trainer decide se usa shift)
        return float(total_reward), behaviour

    # =========================================================
    # Avaliação população (single-env)
    # =========================================================
    def evaluate_population(self, env_factory: Callable[[], Enviroment], max_steps, episodes_per_individual=3):
        pop = self.ga.population
        N = len(pop)

        # limpar cache de modelos (garante consistência)
        self._model_cache.clear()

        fitnesses = np.zeros(N, dtype=np.float32)
        behaviours = []

        for gi, genome in enumerate(pop):
            total = 0.0
            bcs = []
            for _ in range(episodes_per_individual):
                env = env_factory()
                r, bc = self._evaluate_genome(genome, env, max_steps, genome_id=gi)
                total += r
                bcs.append(bc)

            fitnesses[gi] = total / episodes_per_individual
            behaviours.append(np.mean(bcs, axis=0).astype(np.float32))

        return fitnesses.tolist(), behaviours

    # =========================================================
    # Avaliação população (multi-env)
    # env_factories: lista de factories (funções que retornam ambientes)
    # =========================================================
    def evaluate_population_multi(self, env_factories: List[Callable[[], Enviroment]], max_steps, episodes_per_individual=3):
        pop = self.ga.population
        N = len(pop)

        self._model_cache.clear()

        fitnesses = np.zeros(N, dtype=np.float32)
        behaviours = []

        for gi, genome in enumerate(pop):
            all_rewards = []
            all_bcs = []
            for make_env in env_factories:
                for _ in range(episodes_per_individual):
                    env = make_env()
                    r, bc = self._evaluate_genome(genome, env, max_steps, genome_id=gi)
                    all_rewards.append(r)
                    all_bcs.append(bc)

            fitnesses[gi] = np.mean(all_rewards)
            behaviours.append(np.mean(all_bcs, axis=0).astype(np.float32))

        return fitnesses.tolist(), behaviours

    # =========================================================
    # Reavaliação (múltiplos episódios)
    # =========================================================
    def evaluate_genome_multiple(self, genome, env_factory: Callable[[], Enviroment], max_steps, n_eval=8):
        rewards, bcs = [], []
        for _ in range(n_eval):
            env = env_factory()
            r, bc = self._evaluate_genome(genome, env, max_steps)
            rewards.append(r)
            bcs.append(bc)
        return np.mean(rewards), np.mean(bcs, axis=0)

    # =========================================================
    # TREINO
    # strategy: 'both'|'novelty'|'fitness'
    #   - 'both' usa alpha passado (0..1) como peso para novelty
    #   - 'novelty' força alpha=1.0 (somente novelty)
    #   - 'fitness' força alpha=0.0 (somente fitness)
    # =========================================================
    def train(
        self,
        env_factories,
        max_steps: int,
        generations: int = 50,
        episodes_per_individual: int = 3,
        alpha: float = 0.7,
        strategy: str = "both",
        verbose: bool = True,
        champion_eval_episodes: int = 8,
        external_generation_offset: int = 0
    ):
        history = []

        for gen in range(1, generations + 1):
            # avaliar população (multi ou single)
            if isinstance(env_factories, list):
                fitnesses, behaviours = self.evaluate_population_multi(
                    env_factories, max_steps, episodes_per_individual
                )
            else:
                fitnesses, behaviours = self.evaluate_population(
                    env_factories, max_steps, episodes_per_individual
                )

            fitnesses_np = np.array(fitnesses, dtype=np.float32)

            best_idx = int(np.argmax(fitnesses_np))

            # validação mais rigorosa do melhor candidato (múltiplos episódios)
            cand_score, _ = self.evaluate_genome_multiple(
                self.ga.population[best_idx],
                (env_factories if callable(env_factories) else env_factories),
                max_steps,
                n_eval=champion_eval_episodes
            )

            if cand_score > self.best_score:
                self.best_score = cand_score
                self.best_genome = self.ga.population[best_idx].copy()
                if verbose:
                    print(f"--> Novo BEST global (gen {external_generation_offset + gen}) score={cand_score:.4f}")

            # definir alpha de acordo com strategy
            if strategy == "novelty":
                use_alpha = 1.0
            elif strategy == "fitness":
                use_alpha = 0.0
            else:
                use_alpha = float(alpha)

            # faz a evolução (GA interno decide mutação / crossover)
            mean_nov, max_nov = self.ga.evolve(behaviours, fitnesses, alpha=use_alpha)

            history.append({
                "generation": external_generation_offset + gen,
                "mean_fitness": float(fitnesses_np.mean()),
                "max_fitness": float(fitnesses_np.max()),
                "mean_novelty": float(mean_nov),
                "max_novelty": float(max_nov),
                "best_idx": best_idx,
                "best_score_global": float(self.best_score),
                "strategy": strategy,
                "alpha_used": use_alpha
            })

            if verbose:
                print(
                    f"[Gen {external_generation_offset + gen}] fitness: mean={fitnesses_np.mean():.4f} "
                    f"max={fitnesses_np.max():.4f} | novelty: mean={mean_nov:.4f} max={max_nov:.4f} | best_idx={best_idx}"
                )

        return history

    # =========================================================
    # Utility
    # =========================================================
    def get_champion_agent(self):
        genome = self.best_genome or self.ga.population[0]
        model = self.model_builder()

        set_weights_vector(model, genome)

        agent = EvolvedAgent(id="champion", model=model)
        agent.set_genoma(genome)

        return agent

    def save_champion(self, path, env_factory, max_steps, n_eval=10, threshold=None):
        if self.best_genome is None:
            raise RuntimeError("No best genome available")

        score, _ = self.evaluate_genome_multiple(
            self.best_genome, env_factory, max_steps, n_eval
        )

        if threshold is not None and score < threshold:
            return False, score

        # salvar genoma como ficheiro numpy (.npy)
        np.save(path, self.best_genome)

        return True, score
