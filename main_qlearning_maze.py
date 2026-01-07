# main_qlearning_maze.py
import sys
import numpy as np
from simulator.simulator import Simulator
from environments.environment_maze import MazeEnv
from agents.qlearning_agent import QLearningAgent


# ============================================
# Função para testar usando o mesmo protocolo do treino
# ============================================
def test_agent_same_protocol(agent, env, max_steps=1000, render=False):
    """
    Testa o agente usando o mesmo protocolo do train_episode.
    Isso garante que o comportamento seja consistente com o treino.
    """
    from environments.renderer import EnvRenderer
    
    env.reset()
    agent.reset()
    
    # Garantir epsilon = 0 para teste puro (sem exploração)
    agent.epsilon = 0.0
    
    h = env.tamanho[1]
    start_pos = (1, h - 1)
    env.regista_agente(agent, start_pos)
    agent.posicao = start_pos
    
    obs = env.observacaoPara(agent)
    agent.observacao(obs)
    
    bx, by = env.goal_pos
    total_reward = 0.0
    steps = 0
    done = False
    prev_dist = np.hypot(start_pos[0] - bx, start_pos[1] - by)
    visited = {start_pos}
    pos_now = start_pos
    
    # Renderer opcional
    renderer = None
    if render:
        renderer = EnvRenderer(env)
    
    trajectory = [start_pos]
    
    while steps < max_steps and not done:
        # Renderizar
        if renderer:
            alive = renderer.draw(env.posicoes_agentes)
            if not alive:
                break
        
        state = obs
        action = agent.age()
        reward, done, info = env.agir(action, agent)
        
        pos_now = env.get_posicao_agente(agent) or pos_now
        x, y = pos_now
        
        # Reward shaping (igual ao treino, mas apenas para visualização)
        # NOTA: Não atualizamos Q-table durante teste
        reward_shaped = reward
        if True:  # Sempre aplicar shaping para consistência visual
            if pos_now not in visited:
                new_dist = np.hypot(x - bx, y - by)
                delta = max(0, prev_dist - new_dist)
                reward_shaped += delta * 5.0
                reward_shaped += 10.0
                visited.add(pos_now)
                prev_dist = new_dist
            reward_shaped -= 0.5  # Custo por passo
        
        if info.get("collision", False):
            reward_shaped -= 10.0
        
        if pos_now == (bx, by) or info.get("reached_beacon", False):
            reward_shaped += 1000.0
            done = True
        
        total_reward += reward_shaped
        trajectory.append(pos_now)
        
        next_obs = env.observacaoPara(agent) if not done else None
        if next_obs is not None:
            agent.observacao(next_obs)
        
        obs = next_obs
        
        if hasattr(env, "atualizacao"):
            env.atualizacao()
        
        steps += 1
    
    if renderer:
        renderer.close()
    
    final_pos = env.get_posicao_agente(agent) or pos_now
    reached_goal = (final_pos == (bx, by))
    
    return {
        "total_reward": total_reward,
        "steps": steps,
        "reached_goal": reached_goal,
        "final_pos": final_pos,
        "goal_pos": (bx, by),
        "trajectory": trajectory
    }


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Carregar agente Q-Learning treinado
    path = "model/best_agent_qlearning_maze.pkl"
    
    try:
        agent = QLearningAgent.load(path, id="qlearning_loaded")
        print("Agente Q-Learning carregado com sucesso!")
        print(f"Tamanho da Q-table: {len(agent.q_table)} estados")
        print(f"Epsilon original: {agent.epsilon:.4f}")
    except Exception as e:
        print(f"Erro ao carregar agente: {e}")
        print("Certifique-se de que o agente foi treinado primeiro (train_qlearning_maze.py)")
        sys.exit(1)
    
    # Criar ambiente Maze (pode escolher a dificuldade)
    dificuldade = 2  # 0=fácil, 1=médio, 2=difícil
    env = MazeEnv(dificuldade=dificuldade, max_steps=1000)
    
    print(f"\nAmbiente criado:")
    print(f"  - Dificuldade: {dificuldade}")
    print(f"  - Tamanho: {env.tamanho}")
    print(f"  - Goal: {env.goal_pos}")
    print(f"  - Max steps: 1000")
    
    # Testar múltiplas vezes para ver taxa de sucesso
    print("\n" + "="*60)
    print("TESTE - Mesmo protocolo do treino")
    print("="*60)
    
    num_tests = 5
    successes = 0
    total_rewards = []
    
    for i in range(num_tests):
        res = test_agent_same_protocol(agent, env, max_steps=1000, render=(i == 0))
        total_rewards.append(res['total_reward'])
        if res['reached_goal']:
            successes += 1
            print(f"Teste {i+1}: ✓ Sucesso! Steps: {res['steps']}, Reward: {res['total_reward']:.2f}")
        else:
            print(f"Teste {i+1}: ✗ Falhou. Steps: {res['steps']}, Final: {res['final_pos']}, Reward: {res['total_reward']:.2f}")
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"Taxa de sucesso: {successes}/{num_tests} ({100*successes/num_tests:.1f}%)")
    print(f"Reward médio: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Epsilon usado: 0.0 (exploração desligada)")
    print("="*60)
    
    # Teste adicional usando Simulator (para comparação)
    print("\n" + "="*60)
    print("TESTE - Usando Simulator (protocolo diferente)")
    print("="*60)
    sim = Simulator(env, max_steps=1000)
    sim.agentes[agent.id] = agent
    
    env.reset()
    agent.reset()
    agent.epsilon = 0.0  # Garantir sem exploração
    start = (1, env.tamanho[1] - 1)
    env.regista_agente(agent, start)
    obs = env.observacaoPara(agent)
    agent.observacao(obs)
    
    res_sim = sim.run_episode(render=False)
    print(f"Simulator - Reached Goal: {res_sim['reached_goal']}")
    print(f"Simulator - Steps: {res_sim['steps']}")
    print(f"Simulator - Final: {res_sim['final_pos']}")
    print("="*60)
