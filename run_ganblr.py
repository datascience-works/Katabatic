from katabatic.artifacts import LocalArtifactStore
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

# Create an artifact store
store = LocalArtifactStore("artifacts")

# Create the pipeline
pipeline = TrainTestSplitPipeline(model=GANBLR())

# Run GANBLR
results = pipeline.run(
    input_csv="preprocessed_data/car.csv",
    dataset_name="car",
    artifact_store=store,
)

print("\n===== Pipeline Finished =====")
print(results)