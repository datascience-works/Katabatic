from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    dataset_path="benchmarks/splits/adult/train_full.csv",
    synthetic_path="benchmarks/synthetic/adult/synthpop/synthetic.csv",
)

print("SynthPop Adult generation completed.")
