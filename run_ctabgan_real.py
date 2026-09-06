from katabatic.models.ctabgan_plus.models import CTABGANPlus

model = CTABGANPlus(config={"epochs": 200})

model.train(
    dataset_dir="sample_data/magic",
    synthetic_dir="synthetic/magic/ctabgan_plus",
    categorical=[10]
)

X_synth = model.sample(1000)
print("Training complete.")
print(X_synth.head() if hasattr(X_synth, "head") else X_synth[:5])
