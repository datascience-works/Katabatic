import os
import sys
import warnings
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

from katabatic.experimental.models.tabddpm.models import (  # noqa: E402
    _TABDDPM_EXTERNAL,
    Tabddpm,
)

config = RunConfig(
    dataset_name="adult",
    model_name="tabddpm",
    categorical_cols=[
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
    target_col_raw="class",
    constraints={
        "age": (17, 90),  # working age range
        "fnlwgt": (12285, 1490400),  # census sampling weight, dataset min/max
        "capital-gain": (0, 99999),  # cannot be negative, capped at 99999 in dataset
        "capital-loss": (0, 4356),  # cannot be negative, capped at 4356 in dataset
        "hours-per-week": (1, 99),  # at least 1 hour, max 99 in dataset
    },
)


def main():
    if not _TABDDPM_EXTERNAL:
        warnings.warn(
            "The local TabDDPM fallback rounds categorical predictions in its loss, "
            "yielding zero feature-learning gradients. Use the full TabDDPM backend "
            "for meaningful categorical benchmark results.",
            RuntimeWarning,
            stacklevel=2,
        )
    train_df, test_df, target_col, paths = preprocess_and_split(config)

    print("\n" + "=" * 60)
    print("STEP 3 - Train TabDDPM")
    print("=" * 60)
    features = train_df.drop(columns=[target_col]).copy()
    # The numerical sampler operates on a bounded scale. Fit only on training
    # features and restore their original scale before saving or evaluation.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features[config.continuous_cols] = scaler.fit_transform(
        features[config.continuous_cols].astype(float)
    )
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
        normalized = (
            synthetic_df[config.continuous_cols]
            .astype(float)
            .clip(
                lower=features[config.continuous_cols].min(),
                upper=features[config.continuous_cols].max(),
                axis="columns",
            )
        )
        synthetic_df[config.continuous_cols] = scaler.inverse_transform(normalized)
        # Guard the inverse transform's floating-point roundoff at the bounds.
        synthetic_df[config.continuous_cols] = synthetic_df[
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
