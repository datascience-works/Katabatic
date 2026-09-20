from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    "benchmarks/splits/magic",
    synthetic_dir="benchmarks/synthetic/magic/synthpop",
)

print("SynthPop Magic generation completed.")
