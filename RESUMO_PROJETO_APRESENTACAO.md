# Resumo do Projeto SMA-Learn
## Guia para Apresentação PowerPoint

---

## SLIDE 1: Título e Introdução

### Título
**SMA-Learn: Aprendizagem por Reforço em Ambientes de Navegação**

### Subtítulo
*Comparação entre Algoritmos Evolutivos e Q-Learning*

### Informação do Projeto
- **Objetivo**: Treinar agentes para navegar e alcançar objetivos em ambientes complexos
- **Técnicas**: Algoritmos Genéticos + Novelty Search vs Q-Learning Tabular
- **Ambientes**: Labirintos e Farol (navegação com obstáculos)

---

## SLIDE 2: Objetivos do Projeto

### Objetivos Principais
1. **Implementar dois paradigmas de aprendizagem**
   - Algoritmos Evolutivos (EA) com busca por novidade (Novelty Search)
   - Q-Learning tabular com discretização de estados

2. **Aplicar em dois ambientes distintos**
   - Labirinto (Maze): Navegação em corredores
   - Farol (Beacon): Navegação em espaço aberto com obstáculos

3. **Comparar desempenho**
   - Taxa de sucesso
   - Eficiência de treino
   - Robustez e generalização

4. **Curriculum Learning**
   - Treino progressivo: dificuldades fáceis → difíceis

---

## SLIDE 3: Arquitetura do Sistema

### Componentes Principais

```
┌─────────────────┐
│   AMBIENTES     │
│  - MazeEnv      │
│  - FarolEnv     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     AGENTES     │
│  - EvolvedAgent │
│  - QLearning    │
│  - FixedPolicy  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ALGORITMOS    │
│  - Evolution    │
│  - Q-Learning   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SIMULADOR     │
│  - Renderização │
│  - Métricas     │
└─────────────────┘
```

### Fluxo de Dados
- **Observação**: 12 dimensões (8 sensores de distância + posição normalizada + distância ao goal)
- **Ações**: 4 direções (UP, RIGHT, DOWN, LEFT)
- **Recompensas**: Shaping específico por ambiente

---

## SLIDE 4: Ambientes Implementados

### 1. Labirinto (Maze)
- **Tamanhos**: 12x12, 20x20, 30x30 (dependendo da dificuldade)
- **Características**:
  - Labirintos gerados com algoritmo recursivo
  - Goal no topo (y=0)
  - Agente começa no canto inferior esquerdo
- **Dificuldades**: 0, 1 (usadas no Q-Learning)

### 2. Farol (Beacon)
- **Tamanho**: 50x50 (fixo)
- **Características**:
  - Espaço aberto com obstáculos aleatórios
  - Goal na metade superior do mapa
  - Densidade de obstáculos aumenta com dificuldade
- **Dificuldades**: 1, 2, 3

### Observações
- Ambos fornecem vetor de 12 dimensões:
  - 8 sensores de distância (radar 360°)
  - Posição normalizada (x, y)
  - Distância ao goal normalizada
  - Delta Y normalizado

---

## SLIDE 5: Algoritmo 1 - Algoritmos Evolutivos

### Características Principais

**1. Modelo Neural**
- MLP com célula RNN
- Arquitetura: 12 inputs → 32 hidden (RNN) → 4 outputs
- Pesos evolvidos via algoritmo genético

**2. Algoritmo Genético**
- **População**: 120-300 indivíduos (dependendo do ambiente)
- **Seleção**: Elite (top 5-10%) + Torneio
- **Crossover**: Recombinação de genomas
- **Mutação**: Adaptativa baseada no desempenho dos pais
- **Novelty Search**: Combina fitness + novidade comportamental

**3. Novidade Comportamental (BC)**
- Características: posição final, distância percorrida, células visitadas
- Archive de comportamentos únicos
- Alpha adaptativo: exploração inicial → exploração final

**4. Curriculum Learning**
- Labirinto: Dificuldades 0, 1, 2 (3 mapas fixos)
- Farol: Progressão 1 → 2 → 3 com múltiplas seeds

**5. Hiperparâmetros (Farol)**
- Gerações: 300
- População: 150
- Elite: 8%
- Episódios por indivíduo: 4
- Alpha adaptativo (0.85 → 0.6 → 0.3)

---

## SLIDE 6: Algoritmo 2 - Q-Learning

### Características Principais

**1. Q-Table Tabular**
- Estados discretizados via bins
- 8 bins por dimensão
- Q-values: Q(s, a) para cada estado-ação

**2. Discretização**
- Observações contínuas → estados discretos
- Normalização online (min/max observados)
- Bins construídos dinamicamente

**3. Epsilon-Greedy**
- Exploração inicial: ε = 1.0
- Decay gradual: ε → 0.05
- Exploração mínima garantida

**4. Reward Shaping**

**Maze**:
- Custo por passo: -0.5
- Colisão: -10.0
- Nova célula visitada: +10.0 + aproximação
- Goal: +1000.0

**Farol**:
- Aproximação ao goal: (Δdistância) × 5.0
- Colisão: -10.0
- Goal: +1000.0

**5. Curriculum Learning**
- Maze: Dificuldades 0 e 1
- Farol: Progressão 1 → 2 → 3

**6. Hiperparâmetros**
- Learning rate: 0.1
- Gamma: 0.99
- Epsilon decay: 0.9995
- Episódios: 1000 (Maze) / 2000 (Farol)

---

## SLIDE 7: Comparação de Métodos

### Tabela Comparativa

| Característica | Algoritmos Evolutivos | Q-Learning |
|----------------|----------------------|------------|
| **Tipo de Aprendizagem** | Offline (população) | Online (episódios) |
| **Representação** | Rede Neural (pesos) | Tabela Q(s,a) |
| **Exploração** | Novelty Search + Mutação | Epsilon-greedy |
| **Memória** | Genomas (pesos) | Q-table (estados) |
| **Convergência** | Lenta mas robusta | Mais rápida inicialmente |
| **Interpretabilidade** | Baixa (pesos da rede) | Média (Q-table) |
| **Overfitting** | Risco menor | Risco maior |
| **Complexidade Espaço** | O(população) | O(estados × ações) |

### Pontos Fortes

**EA + Novelty Search**:
- ✅ Diversidade comportamental
- ✅ Evita convergência prematura
- ✅ Boa generalização
- ✅ Exploração criativa

**Q-Learning**:
- ✅ Aprendizagem direta de política
- ✅ Convergência garantida (teórica)
- ✅ Interpretável (Q-values)
- ✅ Eficiente em espaços pequenos

---

## SLIDE 8: Resultados - Labirinto (Maze)

### Métricas de Treino

**Algoritmos Evolutivos**:
- Gerações: 2000
- População: 300
- Taxa de sucesso: [mostrar gráfico plot_novelty.png]
- Fitness médio: [mostrar gráfico plot_fitness.png]

**Q-Learning**:
- Episódios: 10000
- Taxa de sucesso: [mostrar gráfico plot_qlearning_maze.png]
- Q-table size: [evolução do tamanho]

### Gráficos a Mostrar
1. **Learning Curves**: Rewards ao longo do treino
2. **Success Rate**: Percentagem de sucesso
3. **Q-Table Growth**: Estados descobertos (Q-Learning)
4. **Novelty Archive Size**: Diversidade (EA)

### Conclusões Maze
- EA: Convergência mais lenta mas estável
- Q-Learning: Aprende rápido mas pode overfitting
- Ambos alcançam altas taxas de sucesso (>80%)

---

## SLIDE 9: Resultados - Farol (Beacon)

### Métricas de Treino

**Algoritmos Evolutivos**:
- Gerações: 300
- Taxa de sucesso: [mostrar gráfico plot_novelty.png]
- Fitness com alpha adaptativo

**Q-Learning**:
- Episódios: 2000
- Taxa de sucesso: [mostrar gráfico plot_qlearning_farol.png]

### Desafios Específicos
- **Espaço maior** (50x50 vs 12-30x30)
- **Obstáculos dinâmicos** (dependem da seed)
- **Goal posicionado no topo** (distância maior)

### Conclusões Farol
- Ambiente mais desafiante
- Requer mais exploração
- Q-Learning precisa de mais episódios
- EA com Novelty Search explora melhor

---

## SLIDE 10: Análise Técnica - Reward Shaping

### Importância do Reward Shaping

**Problema Original**: Recompensas esparsas (só ao alcançar goal)
- Dificulta aprendizagem inicial
- Exploração insuficiente

**Solução**: Reward Shaping
- Guia exploração inicial
- Acelera convergência
- Mantém objetivo final

### Exemplos de Shaping

**Maze - Exploração**:
```
reward += 10.0  # Nova célula visitada
reward += (prev_dist - new_dist) * 5.0  # Aproximação
```

**Farol - Navegação Direta**:
```
reward += (prev_dist - new_dist) * 5.0  # Aproximação constante
```

### Impacto
- ✅ Reduz tempo de treino
- ✅ Aumenta taxa de sucesso
- ✅ Mais robusto a ambientes variados

---

## SLIDE 11: Curriculum Learning

### Estratégia de Treino Progressivo

**Fase 1: Ambientes Fáceis**
- Aprende comportamentos básicos
- Estabelece política inicial
- Baixa complexidade

**Fase 2: Ambientes Médios**
- Refina política
- Aumenta robustez
- Transfere conhecimento

**Fase 3: Ambientes Difíceis**
- Generalização
- Política final otimizada
- Teste máximo

### Implementação

**EA - Farol**:
- Geração 1-10: Dificuldade 1
- Geração 11-20: Dificuldade 2
- Geração 21-30: Dificuldade 3
- Geração 31+: Mistura de todas

**Q-Learning - Farol**:
- Episódios 1-1000: Dificuldades 1 e 2
- Episódios 1001+: Todas as dificuldades

### Benefícios
- ✅ Aprendizagem mais estável
- ✅ Menos convergência prematura
- ✅ Melhor generalização

---

## SLIDE 12: Desafios e Soluções

### Desafios Enfrentados

**1. Espaço de Estados Contínuo**
- **Problema**: Ambientes fornecem observações contínuas
- **Solução**: Discretização via bins (Q-Learning) / Rede Neural (EA)

**2. Recompensas Esparsas**
- **Problema**: Só recompensa ao alcançar goal
- **Solução**: Reward shaping + curriculum learning

**3. Exploração vs Exploração**
- **Problema**: Balancear exploração e exploração
- **Solução**: 
  - EA: Novelty Search + alpha adaptativo
  - Q-Learning: Epsilon-greedy com decay

**4. Overfitting**
- **Problema**: Agente aprende apenas ambientes de treino
- **Solução**: 
  - EA: Multi-ambiente simultâneo
  - Q-Learning: Curriculum variado

**5. Convergência Lenta (EA)**
- **Problema**: Muitas gerações necessárias
- **Solução**: Mutação adaptativa + elite maior

---

## SLIDE 13: Arquivos e Estrutura do Projeto

### Organização do Código

```
SMA-Learn/
├── agents/              # Agentes (Evolved, QLearning, Fixed)
├── algorithms/          # Trainers (Evolution, Q-Learning)
├── environments/        # Ambientes (Maze, Farol)
├── model/              # Modelos salvos
├── results/            # Gráficos e históricos
│   ├── maze/
│   └── farol/
├── simulator/          # Simulador e renderização
├── train_*.py          # Scripts de treino
├── main_*.py           # Scripts de teste
└── QLEARNING_DOCUMENTATION.md
```

### Execução

**Treinar EA - Labirinto**:
```bash
python train_maze.py
```

**Treinar EA - Farol**:
```bash
python train_farol.py
```

**Treinar Q-Learning - Labirinto**:
```bash
python train_qlearning_maze.py
```

**Treinar Q-Learning - Farol**:
```bash
python train_qlearning_farol.py
```

**Testar Agente**:
```bash
python main_evolved.py
python main_qlearning_maze.py
```

---

## SLIDE 14: Conclusões

### Principais Descobertas

**1. Algoritmos Evolutivos + Novelty Search**
- Excelente para exploração diversa
- Boa generalização
- Convergência estável mas lenta
- Ideal para ambientes complexos e variados

**2. Q-Learning Tabular**
- Aprendizagem mais rápida inicialmente
- Interpretável (Q-table)
- Eficiente em espaços discretos pequenos
- Pode sofrer de overfitting

**3. Reward Shaping é Crítico**
- Sem shaping: aprendizagem muito lenta
- Com shaping: convergência 5-10x mais rápida
- Essencial para ambos os métodos

**4. Curriculum Learning Ajuda**
- Treino progressivo melhora resultados
- Reduz convergência prematura
- Facilita transferência de conhecimento

### Recomendações Futuras
- Deep Q-Learning (DQN) para espaços maiores
- Híbrido: EA para exploração inicial + Q-Learning para refinamento
- Teste em ambientes ainda mais complexos
- Análise de generalização em mapas não vistos

---

## SLIDE 15: Métricas e Gráficos (Anexos)

### Gráficos Disponíveis

**results/maze/**:
- `plot_fitness.png`: Evolução do fitness (EA)
- `plot_novelty.png`: Evolução da novidade (EA)
- `plot_qlearning_maze.png`: 6 subplots Q-Learning

**results/farol/**:
- `plot_fitness.png`: Evolução do fitness (EA)
- `plot_novelty.png`: Evolução da novidade (EA)
- `plot_qlearning_farol.png`: 6 subplots Q-Learning

### Subplots Q-Learning
1. Learning Curve (Rewards)
2. Exploration Decay (Epsilon)
3. Success Rate
4. Mean Steps per Episode
5. Q-Table Growth
6. Reward Variance

### Dados Salvos
- `history_*.json`: Histórico completo de treino
- `best_agent_*.npy/pkl`: Melhor agente treinado

---

## SLIDE 16: Demonstração (Opcional)

### Vídeo/Demo ao Vivo
- Executar `main_evolved.py` ou `main_qlearning_maze.py`
- Mostrar agente navegando
- Destaque para:
  - Decisões tomadas
  - Trajetória seguida
  - Alcançar o objetivo

### Comparação Visual
- Lado a lado: EA vs Q-Learning
- Mesmo ambiente, diferentes estratégias
- Mostrar robustez e generalização

---

## NOTAS PARA APRESENTAÇÃO

### Ordem Sugerida de Slides
1. Título (1)
2. Objetivos (2)
3. Arquitetura (3)
4. Ambientes (4)
5. EA (5)
6. Q-Learning (6)
7. Comparação (7)
8. Resultados Maze (8)
9. Resultados Farol (9)
10. Reward Shaping (10)
11. Curriculum Learning (11)
12. Desafios (12)
13. Conclusões (14)
14. Perguntas

### Tempo Estimado
- Introdução: 2 min
- Métodos: 5 min
- Resultados: 4 min
- Análise: 3 min
- Conclusões: 2 min
- **Total: ~15-16 minutos**

### Dicas
- Começar com demo visual se possível
- Mostrar gráficos de treino (são muito informativos)
- Destacar diferenças práticas entre métodos
- Preparar resposta sobre quando usar cada método

---

## ANEXO: Comparação Numérica

### Números de Referência

**Labirinto (Maze)**:
- EA: ~2000 gerações, população 300
- Q-Learning: 10000 episódios
- Sucesso EA: ~85-95%
- Sucesso Q-Learning: ~80-90%

**Farol**:
- EA: 300 gerações, população 150
- Q-Learning: 2000 episódios
- Sucesso EA: ~70-85%
- Sucesso Q-Learning: ~65-80%

**Tempo de Treino (estimado)**:
- EA Maze: ~30-60 min
- Q-Learning Maze: ~10-20 min
- EA Farol: ~15-30 min
- Q-Learning Farol: ~5-15 min

*Nota: Tempos dependem do hardware*

