import pandas as pd


def get_column_distribution(series: pd.Series):
    """
    Get empirical values and probabilities for a column.
    """
    vc = series.value_counts(normalize=True, dropna=False)

    values = vc.index.to_numpy()
    probs = vc.values.astype(float)

    probs = probs / probs.sum()

    return values, probs


def sample_column(rng, values, probs, n_rows: int):
    """
    Sample values from a column distribution.
    """
    return rng.choice(
        values,
        size=n_rows,
        p=probs,
    )