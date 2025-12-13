# Q-Learning - Documentação

## O que é Q-Learning?

Q-Learning é um algoritmo de aprendizagem por reforço (Reinforcement Learning) que permite a um agente aprender uma política ótima através de exploração e exploração do ambiente. É um algoritmo **off-policy** e **model-free**, o que significa que aprende a melhor ação a tomar em cada estado sem necessitar de um modelo do ambiente.

### Conceitos Fundamentais

#### 1. Q-Table
A Q-table é uma tabela que armazena o valor esperado (Q-value) de tomar uma ação `a` em um estado `s`. Representado como Q(s,a), indica a recompensa total esperada desde o estado atual até ao fim do episódio.

#### 2. Equação de Bellman (Q-Learning)
A atualização do Q-value segue a equação:

```
Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

Onde:
- **α (learning_rate)**: Taxa de aprendizagem (0.1 por padrão)
- **γ (gamma)**: Fator de desconto (0.99 por padrão)
- **r**: Recompensa recebida
- **s'**: Próximo estado
- **max(Q(s',a'))**: Melhor Q-value no próximo estado

#### 3. Política Epsilon-Greedy
Para balancear exploração e exploração:
- Com probabilidade **ε (epsilon)**: escolhe ação aleatória (exploração)
- Com probabilidade **1-ε**: escolhe ação com maior Q-value (exploração)

O epsilon começa em 1.0 (exploração total) e decai gradualmente durante o treino.

#### 4. Discretização de Estados
Como os ambientes fornecem observações contínuas (12 dimensões), é necessário discretizar os estados:
- Cada dimensão é dividida em **bins** (8 bins por padrão)
- Um estado contínuo é mapeado para uma tupla de índices de bins
- Exemplo: `(3, 5, 2, 7, ...)` representa um estado discretizado

## Implementação no Projeto

### Estrutura de Ficheiros

- **`agents/qlearning_agent.py`**: Agente Q-Learning com Q-table tabular
- **`algorithms/qlearning_trainer.py`**: Trainer que gerencia o processo de treino
- **`train_qlearning_maze.py`**: Script de treino para o ambiente Maze
- **`train_qlearning_farol.py`**: Script de treino para o ambiente Farol
- **`main_qlearning_maze.py`**: Script de teste para o ambiente Maze
- **`main_qlearning_farol.py`**: Script de teste para o ambiente Farol

### Componentes Principais

#### QLearningAgent

O agente mantém:
- **Q-table**: Dicionário `{state_tuple: np.array([Q(s,a0), Q(s,a1), Q(s,a2), Q(s,a3)])}`
- **Bins de discretização**: Construídos online baseados nas observações mínimas/máximas
- **Epsilon**: Controla o balanço exploração/exploração

Métodos principais:
- `age()`: Escolhe ação usando epsilon-greedy
- `update_q_value()`: Atualiza Q-value usando equação de Bellman
- `_discretize_state()`: Converte observação contínua em estado discreto
- `save()/load()`: Persistência da Q-table e parâmetros

#### QLearningTrainer

Gerencia o processo de treino:
- Executa episódios de treino
- Aplica reward shaping específico para cada ambiente
- Realiza avaliação periódica (sem exploração)
- Seleciona e guarda o melhor agente baseado em success rate

### Reward Shaping

#### Maze
- Custo por passo: -0.5
- Colisão: -10.0
- Nova célula visitada: +10.0 + recompensa baseada em aproximação
- Goal alcançado: +1000.0

#### Farol
- Recompensa baseada em distância: `+ (prev_dist - new_dist) * 5.0`
- Colisão: -10.0
- Goal alcançado: +1000.0

### Curriculum Learning

Para facilitar a aprendizagem, o treino usa curriculum learning:

**Maze**:
- Episódios 1-1000: Dificuldades 0 e 1 (12x12 e 20x20)
- Episódios 1001+: Todas as dificuldades (inclui 30x30)

**Farol**:
- Episódios 1-1000: Dificuldades 1 e 2
- Episódios 1001+: Todas as dificuldades (1, 2, 3)

## Como Treinar

### Maze

```bash
python train_qlearning_maze.py
```

**Parâmetros principais**:
- Episódios: 1000
- Max steps: 1000
- Learning rate: 0.1
- Gamma: 0.99
- Epsilon: 1.0 → 0.05 (decay lento)
- N_bins: 8

**Resultados guardados**:
- `model/best_agent_qlearning_maze.pkl`: Melhor agente
- `results/maze/history_qlearning_maze.json`: Histórico de treino
- `results/maze/plot_qlearning_maze.png`: Gráficos de aprendizagem

### Farol

```bash
python train_qlearning_farol.py
```

**Parâmetros principais**:
- Episódios: 2000
- Max steps: 350
- Mesmos hiperparâmetros do Maze

**Resultados guardados**:
- `model/best_agent_qlearning_farol.pkl`: Melhor agente
- `results/farol/history_qlearning_farol.json`: Histórico de treino
- `results/farol/plot_qlearning_farol.png`: Gráficos de aprendizagem

## Como Testar

### Maze

```bash
python main_qlearning_maze.py
```

O script:
1. Carrega o melhor agente treinado
2. Define epsilon=0.0 (sem exploração)
3. Executa múltiplos testes (5 execuções)
4. Mostra taxa de sucesso e estatísticas

### Farol

```bash
python main_qlearning_farol.py
```

Mesmo protocolo do Maze, adaptado para o ambiente Farol.

## Seleção do Melhor Agente

O trainer prioriza agentes que realmente alcançam o objetivo:

1. **Primeiro critério**: Taxa de sucesso (success_rate)
   - Se um agente alcança o objetivo e outro não, o que alcança é melhor
   - Entre agentes que alcançam, o com maior taxa de sucesso é melhor

2. **Segundo critério**: Reward
   - Se a taxa de sucesso é igual, escolhe o com maior reward
   - Se nenhum alcançou ainda, usa reward como critério

Isto garante que o agente guardado seja o mais capaz de resolver o problema.

## Gráficos Gerados

Os scripts de treino geram gráficos com 6 subplots:

1. **Learning Curve (Rewards)**: Evolução dos rewards de treino vs avaliação
2. **Exploration Decay**: Evolução do epsilon ao longo do treino
3. **Success Rate**: Percentagem de sucesso em alcançar o goal
4. **Mean Steps**: Média de passos por episódio
5. **Q-Table Growth**: Crescimento do número de estados descobertos
6. **Reward Variance**: Estabilidade do aprendizado (desvio padrão)

## Limitações

- **Q-table tabular**: O espaço de estados cresce exponencialmente com o número de bins
- **Memória**: Pode ocupar muita memória se muitos estados únicos forem descobertos
- **Overfitting**: Pode aprender políticas específicas dos ambientes de treino
- **Discretização fixa**: Bins são fixos após construção inicial

## Vantagens

- **Simplicidade**: Fácil de implementar e entender
- **Interpretabilidade**: Q-table pode ser inspecionada
- **Não requer modelo**: Aprende diretamente da interação com o ambiente
- **Convergência garantida**: Sob certas condições, converge para a política ótima

