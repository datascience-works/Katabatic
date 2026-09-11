from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    dataset_path="benchmarks/splits/shuttle/train_full.csv",
    synthetic_path="benchmarks/synthetic/shuttle/synthpop/synthetic.csv",
)

print("SynthPop Shuttle generation completed.")
