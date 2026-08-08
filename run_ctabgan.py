from katabatic.models.ctabgan_plus.models import CopulaGANModel

# Workaround: DATASET_LABEL_MAP is out of date — actual column is "10", not "class"
CopulaGANModel.DATASET_LABEL_MAP["magic"] = "10"

model = CopulaGANModel(epochs=200, verbose=True)

model.train(
    dataset_dir="sample_data/magic",
    synthetic_dir="synthetic/magic/ctabgan_plus"
)

print("Training complete. Check synthetic/magic/ctabgan_plus/ for x_synth.csv and y_synth.csv")