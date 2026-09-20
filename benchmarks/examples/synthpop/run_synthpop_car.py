from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    "benchmarks/splits/car",
    synthetic_dir="benchmarks/synthetic/car/synthpop",
)

print("SynthPop Car generation completed.")
