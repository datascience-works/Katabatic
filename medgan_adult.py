from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.medgan.models import MEDGAN

# Step 1: Preprocess Adult dataset
discretize_preprocess("raw_data/adult.csv", "discretized_data/adult_bins5.csv",
                      bins=5, strategy="uniform")


# Step 2: Run pipeline with MedGAN
pipeline = TrainTestSplitPipeline(
    model=lambda: MEDGAN(
        gan_epochs=800,       
        ae_pretrain_epochs=50, 
        batch_size=128         
    )
)

pipeline.run(
    input_csv="discretized_data/adult.csv",
    output_dir="outputs/adult",
    synthetic_dir="synthetic/adult",
    real_test_dir="outputs/adult"
)
