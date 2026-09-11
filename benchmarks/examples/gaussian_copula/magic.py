import os
import sys
from time import perf_counter


REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from katabatic.models.gaussian_copula.models import GaussianCopulaModel
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


start_time = perf_counter()

train_csv = os.path.join(
    "sample_data",
    "magic",
    "train_full.csv",
)

test_csv = os.path.join(
    "sample_data",
    "magic",
    "test_full.csv",
)

output_dir = os.path.join(
    "benchmarks",
    "outputs",
    "gaussian_copula",
    "magic",
)

model = GaussianCopulaModel()

pipeline = TrainTestSplitPipeline(model)

result = pipeline.run(
    train_csv=train_csv,
    test_csv=test_csv,
    output_dir=output_dir,
)

end_time = perf_counter()
runtime = end_time - start_time


print("\n" + "=" * 60)
print("Gaussian Copula - Magic Dataset")
print("=" * 60)

print("Pipeline message:", result["message"])
print("Output directory:", result["output_dir"])
print("Synthetic directory:", result["synthetic_dir"])

print("\nTSTR Results:")
print(result["tstr_results"])

print("\nRuntime:")
print(f"{runtime:.2f} seconds")

print("=" * 60)