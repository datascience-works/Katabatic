from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    dataset_path="benchmarks/splits/nursery/train_full.csv",
    synthetic_path="benchmarks/synthetic/nursery/synthpop/synthetic.csv",
)

print("SynthPop Nursery generation completed.")
