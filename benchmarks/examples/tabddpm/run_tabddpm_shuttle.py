import os
import sys
from types import SimpleNamespace

from sklearn.preprocessing import MinMaxScaler

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import (  # noqa: E402
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.experimental.models.tabddpm.models import Tabddpm  # noqa: E402

config = RunConfig(
    dataset_name="shuttle",
    model_name="tabddpm",
    categorical_cols=[],
    continuous_cols=["time", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    target_col_raw="class",
    constraints={},
)


def main():
    train_df, test_df, target_col, paths = preprocess_and_split(config)
    features = train_df.drop(columns=[target_col]).astype(float)
    # The numerical sampler operates on a bounded scale. Fit only on training
    # features and restore their original scale before saving or evaluation.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features.loc[:, config.continuous_cols] = scaler.fit_transform(
        features[config.continuous_cols]
    )

    print("\n" + "=" * 60)
    print("STEP 3 - Train TabDDPM")
    print("=" * 60)
    model = Tabddpm()
    # The array API honours cat_cols. String targets are encoded to contiguous
    # class indices, including datasets whose integer labels start at one.
    model.train(
        features,
        train_df[target_col].astype(str),
        cat_cols=config.categorical_cols,
        config=dict(
            steps=2000,
            num_timesteps=1000,
            batch_size=256,
            d_layers=(256, 256, 256),
            use_ema=True,
            eval_batches=10,
            seed=config.seed,
        ),
    )
    print("\nTabDDPM training complete.")

    def sample_dataframe(n, seed=None):
        if seed is not None:
            model._set_seed(seed)
        synthetic_df = model.sample(n, as_dataframe=True)
        if "label" in synthetic_df.columns and target_col not in synthetic_df.columns:
            synthetic_df = synthetic_df.rename(columns={"label": target_col})
        if not synthetic_df.columns.is_unique or set(synthetic_df.columns) != set(
            train_df.columns
        ):
            raise ValueError(
                "TabDDPM synthetic columns differ from the training schema: "
                f"expected {train_df.columns.tolist()}, got {synthetic_df.columns.tolist()}"
            )
        synthetic_df = synthetic_df.loc[:, train_df.columns].copy()
        normalized = synthetic_df[config.continuous_cols].clip(
            lower=features[config.continuous_cols].min(),
            upper=features[config.continuous_cols].max(),
            axis="columns",
        )
        synthetic_df.loc[:, config.continuous_cols] = scaler.inverse_transform(
            normalized
        )
        # Guard the inverse transform's floating-point roundoff at the bounds.
        synthetic_df.loc[:, config.continuous_cols] = synthetic_df[
            config.continuous_cols
        ].clip(
            lower=train_df[config.continuous_cols].min(),
            upper=train_df[config.continuous_cols].max(),
            axis="columns",
        )
        # TabDDPM's categorical decoders return strings, including numeric bins.
        for column in [*config.categorical_cols, target_col]:
            synthetic_df[column] = synthetic_df[column].astype(train_df[column].dtype)
        return synthetic_df

    print("\n" + "=" * 60)
    print("STEP 4 - Generate synthetic data")
    print("=" * 60)
    synthetic_df = sample_dataframe(len(train_df))
    synthetic_df = save_synthetic(
        synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
    )

    # Stability re-sampling must use the same label and dtype restoration.
    evaluate(
        SimpleNamespace(sample=sample_dataframe),
        config,
        train_df,
        synthetic_df,
        target_col,
        paths,
        test_df,
    )


if __name__ == "__main__":
    main()
