from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    dataset_path="benchmarks/splits/car/train_full.csv",
    synthetic_path="benchmarks/synthetic/car/synthpop/synthetic.csv",
)

print("SynthPop Car generation completed.")
