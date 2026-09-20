from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    "benchmarks/splits/shuttle",
    synthetic_dir="benchmarks/synthetic/shuttle/synthpop",
)

print("SynthPop Shuttle generation completed.")
