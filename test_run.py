from algorithms.train_evolution import train_evolution

if __name__ == "__main__":
    result = train_evolution(
        generations=3,        # pequeno só para testar
        population_size=10,   # pequeno só para ver se funciona
        input_dim=20,         # adapta ao tamanho real da tua observação
        tamanho=(20, 20),
        max_steps=100
    )

    print("\nRESULTADO FINAL DO TESTE:")
    print("Melhor genoma (primeiros 10 valores):", result["melhor_genoma"][:10])
    print("Histórico novelty média:", result["history_mean_nov"])
    print("Histórico novelty máxima:", result["history_max_nov"])