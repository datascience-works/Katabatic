from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    "benchmarks/splits/nursery",
    synthetic_dir="benchmarks/synthetic/nursery/synthpop",
)

print("SynthPop Nursery generation completed.")
