from algorithms.train_evolution import train_evolution

# -----------------------------------------------------
# # AINDA NÃO FOI ALTERADO - VERSÃO ANTIGA
# -----------------------------------------------------

if __name__ == "__main__":
    train_evolution(
        generations=10,
        population_size=50,
        tamanho=(20, 20),
        dificuldade=0,
        max_steps=100
    )