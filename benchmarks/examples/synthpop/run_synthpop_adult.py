from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    "benchmarks/splits/adult",
    synthetic_dir="benchmarks/synthetic/adult/synthpop",
)

print("SynthPop Adult generation completed.")
