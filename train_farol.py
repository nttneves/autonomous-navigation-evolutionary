from algorithms.training_manager import EvolutionTrainingManager
from environments.environment_farol import FarolEnv

def make_env():
    return FarolEnv(tamanho=(21,21), dificuldade=0, max_steps=300)

trainer = EvolutionTrainingManager(
    env_factory=make_env,
    input_dim=10,          
    generations=10,
    population_size=120,
    max_steps=300
)

resultado = trainer.run_training()