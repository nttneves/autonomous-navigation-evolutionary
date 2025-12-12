# algorithms/qlearning_trainer.py
import numpy as np
from typing import Callable, Optional, Tuple
import random
import tempfile
import os
import time

class QLearningTrainer:

    def __init__(
        self,
        input_dim: int = 12,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        n_bins: int = 10,
        seed: int = 42,
        decay_per_step: bool = False,
        min_explore_steps: int = 0,
        reward_clip: Optional[Tuple[float, float]] = (-100.0, 100.0),
        normalize_rewards: bool = False,
        verbose: bool = True
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins

        self.decay_per_step = decay_per_step
        self.min_explore_steps = int(min_explore_steps)
        self.reward_clip = reward_clip
        self.normalize_rewards = normalize_rewards

        np.random.seed(seed)
        random.seed(seed)

        self.best_agent = None
        self.best_score = -np.inf
        self.best_success_rate = 0.0

        self.verbose = bool(verbose)

    def _maybe_clip(self, reward: float) -> float:
        if self.reward_clip is None:
            return reward
        lo, hi = self.reward_clip
        return float(max(lo, min(hi, reward)))

    def _safe_save_agent_copy(self, agent, tmp_prefix="best_agent_"):
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, prefix=tmp_prefix, suffix=".pkl")
            tmp_path = tmp.name
            tmp.close()
            agent.save(tmp_path)
            AgentClass = agent.__class__
            if hasattr(AgentClass, "load"):
                loaded = AgentClass.load(tmp_path, id=f"best_{int(time.time())}")
            else:
                loaded = None
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return loaded
        except Exception as e:
            if tmp is not None:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
            if self.verbose:
                print(f"[QLTrainer] Aviso: não foi possível salvar/copiar agente (erro: {e}).")
            return None

    def train_episode(
        self,
        agent,
        env,
        max_steps: int,
        shaping_reward: bool = True,
        training: bool = True
    ):
        env.reset()
        if hasattr(agent, "reset"):
            agent.reset()

        h = env.tamanho[1]
        start_pos = (1, h - 1)
        if hasattr(env, "regista_agente"):
            env.regista_agente(agent, start_pos)
        else:
            agent.posicao = start_pos

        obs = env.observacaoPara(agent)
        if hasattr(agent, "observacao"):
            agent.observacao(obs)

        bx, by = env.goal_pos
        total_reward = 0.0
        steps = 0
        done = False
        
        px, py = start_pos
        prev_dist = np.hypot(px - bx, py - by)

        is_maze = "maze" in env.__class__.__name__.lower()
        visited = {start_pos}
        pos_now = start_pos

        rewards_list = []

        while steps < max_steps and not done:
            state = obs
            action = agent.age()
            reward, done, info = env.agir(action, agent)

            pos_now = env.get_posicao_agente(agent) or pos_now
            x, y = pos_now

            if shaping_reward:
                try:
                    if is_maze:
                        reward -= 0.5
                        if info.get("collision", False):
                            reward -= 10.0
                        else:
                            new_dist = np.hypot(x - bx, y - by)
                            if pos_now not in visited:
                                delta = max(0, prev_dist - new_dist)
                                reward += delta * 5.0
                                reward += 10.0
                                visited.add(pos_now)
                            prev_dist = new_dist
                        
                        if pos_now == (bx, by) or info.get("reached_beacon", False):
                            reward += 1000.0
                            done = True
                    else:
                        new_dist = np.hypot(x - bx, y - by)
                        reward += (prev_dist - new_dist) * 5.0
                        prev_dist = new_dist
                except Exception as e:
                    if self.verbose:
                        print(f"[QLTrainer] Erro no reward shaping: {e}")
                    pass

            reward = self._maybe_clip(reward)
            total_reward += reward
            rewards_list.append(reward)

            if hasattr(agent, "avaliacaoEstadoAtual"):
                try:
                    agent.avaliacaoEstadoAtual(reward)
                except Exception:
                    pass

            next_obs = env.observacaoPara(agent) if not done else None

            if training and hasattr(agent, "update_q_value"):
                agent.update_q_value(state, action, reward, next_obs, done)

            if hasattr(agent, "observacao") and next_obs is not None:
                agent.observacao(next_obs)

            obs = next_obs
            px, py = x, y

            if hasattr(env, "atualizacao"):
                try:
                    env.atualizacao()
                except Exception:
                    pass

            steps += 1

            if training and self.decay_per_step and hasattr(agent, "decay_epsilon"):
                try:
                    agent.decay_epsilon()
                except Exception:
                    pass

        final_pos = env.get_posicao_agente(agent) or pos_now
        reached_goal = (final_pos == (bx, by))

        if self.normalize_rewards and len(rewards_list) > 0:
            mean_r = np.mean(rewards_list)
            std_r = np.std(rewards_list) + 1e-8
            total_reward = (total_reward - mean_r) / std_r

        return total_reward, steps, reached_goal

    def train(
        self,
        env_factory: Callable[[], object],
        max_steps: int,
        episodes: int = 1000,
        eval_frequency: int = 50,
        eval_episodes: int = 10,
        verbose: Optional[bool] = None
    ):
        if verbose is None:
            verbose = self.verbose

        try:
            from agents.qlearning_agent import QLearningAgent
        except Exception:
            raise RuntimeError("Não foi possível importar QLearningAgent. Verifica a estrutura de ficheiros.")

        agent = QLearningAgent(
            id="qlearning",
            input_dim=self.input_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon=self.epsilon_start,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            n_bins=self.n_bins
        )

        history = []
        episode_rewards = []
        episode_steps = []
        episode_goals = []

        total_steps_counter = 0

        for episode in range(1, episodes + 1):
            env = env_factory()
            reward, steps, reached = self.train_episode(agent, env, max_steps, shaping_reward=True, training=True)

            episode_rewards.append(float(reward))
            episode_steps.append(int(steps))
            episode_goals.append(1 if reached else 0)
            total_steps_counter += steps

            if not self.decay_per_step and hasattr(agent, "decay_epsilon"):
                if total_steps_counter >= self.min_explore_steps:
                    try:
                        agent.decay_epsilon()
                    except Exception:
                        pass

            if episode % eval_frequency == 0 or episode == 1:
                eval_rewards = []
                eval_steps = []
                eval_goals = []

                try:
                    eval_agent = self._safe_save_agent_copy(agent)
                    if eval_agent is None:
                        eval_agent = agent
                        eval_agent_epsilon_backup = getattr(eval_agent, "epsilon", None)
                        try:
                            eval_agent.epsilon = 0.0
                        except Exception:
                            pass
                        use_backup = True
                    else:
                        try:
                            eval_agent.epsilon = 0.0
                        except Exception:
                            pass
                        use_backup = False
                except Exception:
                    eval_agent = agent
                    eval_agent_epsilon_backup = getattr(eval_agent, "epsilon", None)
                    try:
                        eval_agent.epsilon = 0.0
                    except Exception:
                        pass
                    use_backup = True

                for _ in range(max(1, eval_episodes)):
                    eval_env = env_factory()
                    r, s, g = self.train_episode(eval_agent, eval_env, max_steps, shaping_reward=False, training=False)
                    eval_rewards.append(float(r))
                    eval_steps.append(int(s))
                    eval_goals.append(1 if g else 0)

                if 'use_backup' in locals() and use_backup and eval_agent is agent:
                    try:
                        if eval_agent_epsilon_backup is not None:
                            eval_agent.epsilon = eval_agent_epsilon_backup
                    except Exception:
                        pass

                mean_eval_reward = float(np.mean(eval_rewards)) if len(eval_rewards) > 0 else 0.0
                mean_eval_steps = float(np.mean(eval_steps)) if len(eval_steps) > 0 else 0.0
                success_rate = float(np.mean(eval_goals)) * 100.0 if len(eval_goals) > 0 else 0.0

                should_update_best = False
                update_reason = ""
                
                if success_rate > 0.0:
                    if success_rate > self.best_success_rate:
                        should_update_best = True
                        update_reason = f"melhor taxa de sucesso ({success_rate:.1f}% > {self.best_success_rate:.1f}%)"
                    elif success_rate == self.best_success_rate and mean_eval_reward > self.best_score:
                        should_update_best = True
                        update_reason = f"mesma taxa mas melhor reward ({mean_eval_reward:.2f} > {self.best_score:.2f})"
                elif self.best_success_rate == 0.0 and mean_eval_reward > self.best_score:
                    should_update_best = True
                    update_reason = f"melhor reward sem sucesso ({mean_eval_reward:.2f} > {self.best_score:.2f})"

                if should_update_best:
                    self.best_score = mean_eval_reward
                    self.best_success_rate = success_rate
                    if self.verbose:
                        print(f"  --> Novo melhor agente: {update_reason}")
                    
                    best_copy = self._safe_save_agent_copy(agent)
                    if best_copy is not None:
                        try:
                            best_copy.epsilon = 0.0
                        except Exception:
                            pass
                        self.best_agent = best_copy
                    else:
                        try:
                            from agents.qlearning_agent import QLearningAgent
                            self.best_agent = QLearningAgent(
                                id="best",
                                input_dim=self.input_dim,
                                learning_rate=self.learning_rate,
                                gamma=self.gamma,
                                epsilon=0.0,
                                epsilon_min=self.epsilon_min,
                                epsilon_decay=self.epsilon_decay,
                                n_bins=self.n_bins
                            )
                            self.best_agent.q_table = {k: v.copy() for k, v in agent.q_table.items()}
                            if hasattr(agent, 'bins') and agent.bins is not None:
                                self.best_agent.bins = [b.copy() for b in agent.bins]
                                self.best_agent._bins_ready = True
                            if hasattr(agent, 'obs_min'):
                                self.best_agent.obs_min = agent.obs_min.copy()
                                self.best_agent.obs_max = agent.obs_max.copy()
                        except Exception as e:
                            if self.verbose:
                                print(f"[QLTrainer] Aviso: não foi possível criar cópia do melhor agente: {e}")
                            self.best_agent = agent

                recent_slice = eval_frequency if len(episode_rewards) >= eval_frequency else len(episode_rewards)
                recent_rewards = episode_rewards[-recent_slice:] if recent_slice > 0 else episode_rewards
                recent_steps = episode_steps[-recent_slice:] if recent_slice > 0 else episode_steps
                recent_goals = episode_goals[-recent_slice:] if recent_slice > 0 else episode_goals

                mean_reward = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else 0.0
                std_reward = float(np.std(recent_rewards)) if len(recent_rewards) > 0 else 0.0
                mean_steps = float(np.mean(recent_steps)) if len(recent_steps) > 0 else 0.0
                success_rate_train = float(np.mean(recent_goals)) * 100.0 if len(recent_goals) > 0 else 0.0

                history.append({
                    "episode": episode,
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "mean_eval_reward": float(mean_eval_reward),
                    "mean_eval_steps": float(mean_eval_steps),
                    "best_score": float(self.best_score),
                    "best_success_rate": float(self.best_success_rate),
                    "epsilon": float(getattr(agent, "epsilon", 0.0)),
                    "mean_steps": float(mean_steps),
                    "success_rate": float(success_rate),
                    "success_rate_train": float(success_rate_train),
                    "q_table_size": int(len(getattr(agent, "q_table", {})))
                })

                if verbose:
                    print(
                        f"Ep {episode:4d} | Train R: {mean_reward:.2f} ± {std_reward:.2f} | "
                        f"Eval R: {mean_eval_reward:.2f} | Best R: {self.best_score:.2f} | "
                        f"Success: {success_rate:.1f}% | Best Success: {self.best_success_rate:.1f}% | "
                        f"ε: {getattr(agent, 'epsilon', 0.0):.4f} | Q-size: {len(getattr(agent, 'q_table', {}))}"
                    )

        return history, agent

    def save_best_agent(self, path: str):
        if self.best_agent is None:
            raise RuntimeError("Nenhum agente melhor para salvar.")
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        try:
            self.best_agent.save(path)
        except Exception as e:
            try:
                from agents.qlearning_agent import QLearningAgent
                if isinstance(self.best_agent, QLearningAgent):
                    self.best_agent.save(path)
                else:
                    raise
            except Exception as e2:
                raise RuntimeError(f"Falha ao salvar melhor agente: {e} / {e2}")
