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
        """
        Avalia o genoma num ambiente dado. Implementa reward diferente
        para MazeEnv (sem reward por distância) e para FarolEnv (mantém
        a componente de aproximação).
        Também gera um behaviour characterization (BC) com 5 elementos:
          [fx_norm, fy_norm, dist_norm, unique_cells_norm, traj_len_norm]
        """
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

        # distance reference (quando aplicável)
        bx, by = env.goal_pos

        def euclid(pos1, pos2):
            dx = pos1[0] - pos2[0]; dy = pos1[1] - pos2[1]
            return np.sqrt(dx*dx + dy*dy)

        prev_pos = start_pos
        prev_dist = euclid(prev_pos, (bx, by))

        # bookkeeping para BC
        visited = set()
        visited.add(start_pos)
        traj_len = 0

        # identificar tipo de ambiente (não importamos classes para evitar dependências)
        env_name = env.__class__.__name__.lower()
        is_maze = "maze" in env_name or "labir" in env_name  # robusto para MazeEnv
        is_farol = "farol" in env_name or "beacon" in env_name

        while not done and steps < max_steps:
            action = agent.age()
            reward, done, info = env.agir(action, agent)

            # actualizar posição
            pos_now = env.get_posicao_agente(agent)
            if pos_now is None:
                pos_now = prev_pos

            # comportamento de reward:
            if is_maze:
                # Maze: penalização por passo (leve), maior penalização por colisão,
                # grande bónus por chegar ao goal. NÃO usar componente de distância.
                # assumimos que env.agir já penaliza colisões com info["collision"] opcionalmente.
                # normalizamos: keep simple, ajusta coeficientes se necessário.
                step_penalty = -0.01
                collision_penalty = -0.2 if info.get("collision", False) else 0.0
                goal_bonus = 5.0 if info.get("reached_beacon", False) or pos_now == (bx, by) else 0.0

                shaped = step_penalty + collision_penalty + goal_bonus
                reward += shaped

            else:
                # Farol ou outros: mantemos componente de aproximação (shaping).
                new_dist = euclid(pos_now, (bx, by))
                # se env mover de forma válida, recompensa pela aproximação
                reward += (prev_dist - new_dist) * 0.5
                prev_dist = new_dist

            # contabilizar
            agent.avaliacaoEstadoAtual(reward)
            total_reward += reward

            # actualizar trajetoria / visited para BC
            visited.add(pos_now)
            traj_len += 1

            obs = env.observacaoPara(agent)
            agent.observacao(obs)
            prev_pos = pos_now

            if hasattr(env, "atualizacao"):
                env.atualizacao()

            steps += 1

        # BC = posição final normalizada + distância normalizada + coverage + traj_len
        final_pos = env.get_posicao_agente(agent)
        w, h = env.tamanho
        maxd = np.sqrt((w-1)**2 + (h-1)**2)

        if final_pos is None:
            fx_norm = 0.0; fy_norm = 0.0; d_norm = 1.0
        else:
            fx, fy = final_pos
            fx_norm = fx / float(w-1)
            fy_norm = fy / float(h-1)
            d = euclid(final_pos, (bx, by))
            d_norm = d / maxd

        unique_cells_norm = len(visited) / float(w * h)
        traj_len_norm = traj_len / float(max_steps) if max_steps > 0 else 0.0

        behaviour = np.array([
            fx_norm,
            fy_norm,
            d_norm,
            unique_cells_norm,
            traj_len_norm
        ], dtype=np.float32)

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
            # detect whether env_factories is a single callable or a list
            if isinstance(env_factories, list):
                # multi-env evaluation (Curriculum with explicit set)
                fitnesses, behaviours = self.evaluate_population_multi(env_factories, max_steps, episodes_per_individual)
            else:
                fitnesses, behaviours = self.evaluate_population(env_factories, max_steps, episodes_per_individual)

            pop_bcs = [b.copy() for b in behaviours]

            mean_f = float(np.mean(fitnesses))
            max_f = float(np.max(fitnesses))
            best_idx = int(np.argmax(fitnesses))
            best_genome_gen = self.ga.population[best_idx].copy()

            cand_score, _ = self.evaluate_genome_multiple(
                best_genome_gen, (env_factories if isinstance(env_factories, list) else env_factories), max_steps, n_eval=champion_eval_episodes
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