# train_farol.py (simples wrapper)
from algorithms.trainer import EvolutionTrainer
from model.model import create_mlp
from environments.environment_farol import FarolEnv
import json

def make_env():
    return FarolEnv(tamanho=(50,50), dificuldade=1, max_steps=300)

trainer = EvolutionTrainer(model_builder=lambda: create_mlp(input_dim=10),
                           pop_size=80, archive_prob=0.12, elite_fraction=0.05)
history = trainer.train(env_factory=make_env, max_steps=200, generations=20, episodes_per_individual=3, alpha=0.7, verbose=True)

# re-avalia e salva champion (só se a re-avaliação for razoável)
ok, score = trainer.save_champion("model/best_agent_farol.keras", make_env, max_steps=200, n_eval=12, threshold=-0.5)
print("Champion saved:", ok, "score:", score)