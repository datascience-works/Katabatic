from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    dataset_path="benchmarks/splits/magic/train_full.csv",
    synthetic_path="benchmarks/synthetic/magic/synthpop/synthetic.csv",
)

print("SynthPop Magic generation completed.")
