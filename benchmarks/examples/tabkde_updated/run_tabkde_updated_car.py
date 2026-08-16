import random
import numpy as np
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tabkde_updated import TabKDEModel


config = RunConfig(
    dataset_name="car",
    model_name="tabkde_updated",
    categorical_cols=["0", "1", "2", "3", "4", "5"],
    continuous_cols=[],
    target_col_raw="6",
    constraints=None,
)



train_df, test_df, target_col, paths = preprocess_and_split(config)


print("\n" + "=" * 60)
print("STEP 3 — Train TabKDE Updated")
print("=" * 60)

class TabKDEEvaluationAdapter(TabKDEModel):
    def sample(self, n, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        result = super().sample(n)

        if isinstance(result, tuple):
            return result[0]

        return result


model = TabKDEEvaluationAdapter()


model.fit(train_df)

print("\nTabKDE Updated training complete.")


print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)


synthetic_df = model.sample(len(train_df))

synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)



evaluate(
    model,
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)
