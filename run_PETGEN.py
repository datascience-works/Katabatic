from katabatic.artifacts import LocalArtifactStore
from katabatic.models.pategan.models import PATEGAN
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

# Create an artifact store
store = LocalArtifactStore("artifacts")

# Create the pipeline
pipeline = TrainTestSplitPipeline(model=PATEGAN())

# Run PATEGAN
results = pipeline.run(
    input_csv="preprocessed_data/car.csv",
    dataset_name="car",
    artifact_store=store,
)

print("\n===== Pipeline Finished =====")
print(results)