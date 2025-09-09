"""
Inspired by:
https://github.com/opendp/smartnoise-sdk/blob/main/synth/snsynth/transform/minmax.py
and
https://github.com/opendp/smartnoise-sdk/blob/main/synth/snsynth/transform/table.py
"""


import numpy as np

from snsql.sql._mechanisms.approx_bounds import approx_bounds


class MinMaxTransformer:
    """Transforms a column of values to scale between -1.0 and +1.0.

    :param negative: If True, scale between -1.0 and 1.0.  Otherwise, scale between 0.0 and 1.0.
    :param epsilon: The privacy budget to use to infer bounds
    """
    def __init__(self, negative=True, epsilon=0.0):
        self.epsilon = epsilon
        self.negative = negative

    def fit(self, val):
        self.fit_lower, self.fit_upper = approx_bounds(val, self.epsilon)
        if self.fit_lower is None or self.fit_upper is None:
            raise ValueError("MinMaxTransformer could not find bounds.")

    def transform(self, val):
        val = np.clip(val, self.fit_lower, self.fit_upper)
        val = (val - self.fit_lower) / (self.fit_upper - self.fit_lower)
        if self.negative:
            val = (val * 2) - 1
        return val

    def inverse_transform(self, val):
        if self.negative:
            val = (1 + val) / 2
        val = val * (self.fit_upper - self.fit_lower) + self.fit_lower
        return np.clip(val, self.fit_lower, self.fit_upper)


class TableTransformer:
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def fit(self, df):
        self.dtypes = df.dtypes
        self.epsilon_col = self.epsilon / df.shape[1]
        self.transformers = {}

        for col in df.columns:
            self.transformers[col] = MinMaxTransformer(epsilon=self.epsilon_col)
            self.transformers[col].fit(df[col])

    def transform(self, df):
        for col in df.columns:
            df[col] = self.transformers[col].transform(df[col])
        return df

    def inverse_transform(self, df):
        for col in df.columns:
            df[col] = self.transformers[col].inverse_transform(df[col])
        df = df.astype(self.dtypes)
        return df
