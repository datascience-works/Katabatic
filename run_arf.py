from katabatic.models.arf.models import ARFModel

model = ARFModel(num_trees=10, max_iters=10)

model.train(
    data_dir="sample_data/magic",
    synthetic_dir="synthetic/magic/arf"
)

X_synth, y_synth = model.sample(1000)
print("Training complete. Check synthetic/magic/arf/ for x_synth.csv and y_synth.csv")

ks_score = model.evaluate()
print(f"Mean KS statistic: {ks_score}")