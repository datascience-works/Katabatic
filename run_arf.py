from katabatic.models.arf.models import ARFModel

model = ARFModel(num_trees=10, max_iters=10)

model.train(data_dir="sample_data/magic", synthetic_dir="synthetic/magic/arf")

synth_df = model.sample(1000)
print("Training complete. Check synthetic/magic/arf/ for synth_df")

ks_score = model.evaluate()
print(f"Mean KS statistic: {ks_score}")
