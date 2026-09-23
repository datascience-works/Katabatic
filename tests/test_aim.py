from __future__ import annotations

import pandas as pd
import pytest

from katabatic.models.aim import AIMModel
from katabatic.models.aim.utils import (
    decode_dataframe,
    encode_dataframe,
    infer_categorical_columns,
)


def test_aim_initial_state():
    model = AIMModel()

    assert model.is_fitted is False
    assert model.epsilon == 1.0
    assert model.delta == 1e-9
    assert model.degree == 2
    assert model.max_bins == 10
    assert model.max_cells == 10000


def test_aim_invalid_epsilon():
    with pytest.raises(
        ValueError,
        match="epsilon must be greater than zero",
    ):
        AIMModel(epsilon=0)


def test_aim_invalid_delta():
    with pytest.raises(
        ValueError,
        match="delta must be between 0 and 1",
    ):
        AIMModel(delta=1.0)


def test_aim_invalid_max_bins():
    with pytest.raises(
        ValueError,
        match="max_bins must be at least 2",
    ):
        AIMModel(max_bins=1)


def test_aim_required_dependencies():
    assert AIMModel.get_required_dependencies() == [
        "mbi",
    ]


def test_aim_sample_before_training():
    model = AIMModel()

    with pytest.raises(
        RuntimeError,
        match="Call train",
    ):
        model.sample(10)


def test_aim_evaluate_before_training():
    model = AIMModel()

    with pytest.raises(
        RuntimeError,
        match="Call train",
    ):
        model.evaluate()


def test_aim_infer_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["A", "B", "C"],
            "active": [True, False, True],
        }
    )

    result = infer_categorical_columns(df)

    assert result == [
        "city",
        "active",
    ]


def test_aim_resolve_explicit_column_types():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [100.0, 200.0, 300.0],
            "city": ["A", "B", "C"],
        }
    )

    model = AIMModel()

    categorical, continuous = model._resolve_column_types(
        df,
        categorical_cols=["city"],
        continuous_cols=["age", "income"],
    )

    assert categorical == ["city"]
    assert continuous == [
        "age",
        "income",
    ]


def test_aim_resolve_inferred_column_types():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [100.0, 200.0, 300.0],
            "city": ["A", "B", "C"],
        }
    )

    model = AIMModel()

    categorical, continuous = model._resolve_column_types(
        df,
        categorical_cols=None,
        continuous_cols=None,
    )

    assert categorical == ["city"]
    assert continuous == [
        "age",
        "income",
    ]


def test_aim_rejects_overlapping_column_types():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["A", "B", "C"],
        }
    )

    model = AIMModel()

    with pytest.raises(
        ValueError,
        match="both categorical and continuous",
    ):
        model._resolve_column_types(
            df,
            categorical_cols=[
                "city",
                "age",
            ],
            continuous_cols=[
                "age",
            ],
        )


def test_aim_encode_decode_dataframe():
    df = pd.DataFrame(
        {
            "city": [
                "Melbourne",
                "Sydney",
                "Melbourne",
                "Brisbane",
            ],
            "age": [
                20,
                30,
                40,
                50,
            ],
        }
    )

    encoded, encodings, domain_sizes = encode_dataframe(
        df,
        categorical_columns=["city"],
        continuous_columns=["age"],
        max_bins=2,
    )

    assert encoded.shape == df.shape
    assert list(encoded.columns) == [
        "city",
        "age",
    ]

    assert domain_sizes["city"] == 3
    assert domain_sizes["age"] == 2

    assert encoded["city"].dtype.kind in {
        "i",
        "u",
    }

    assert encoded["age"].dtype.kind in {
        "i",
        "u",
    }

    decoded = decode_dataframe(
        encoded,
        encodings,
        [
            "city",
            "age",
        ],
    )

    assert decoded.shape == df.shape
    assert decoded["city"].tolist() == df["city"].tolist()

    assert decoded["age"].tolist() == [
        25,
        25,
        45,
        45,
    ]


def test_aim_encode_rejects_unclassified_columns():
    df = pd.DataFrame(
        {
            "city": ["A", "B"],
            "age": [20, 30],
        }
    )

    with pytest.raises(
        ValueError,
        match="Every training column must be classified",
    ):
        encode_dataframe(
            df,
            categorical_columns=["city"],
            continuous_columns=[],
        )


def test_aim_train_supports_benchmark_column_arguments(
    tmp_path,
    monkeypatch,
):
    import sys
    import types

    class FakeDomain:
        def __init__(self, attrs, shape):
            self.attrs = list(attrs)
            self.shape = list(shape)

        def __iter__(self):
            return iter(self.attrs)

        def __len__(self):
            return len(self.attrs)

        def size(self, clique):
            result = 1

            for column in clique:
                index = self.attrs.index(column)
                result *= self.shape[index]

            return result

    class FakeDataset:
        def __init__(self, df, domain):
            self.df = df
            self.domain = domain

    fake_mbi = types.ModuleType("mbi")
    fake_mbi.Dataset = FakeDataset
    fake_mbi.Domain = FakeDomain
    fake_mbi.FactoredInference = object
    fake_mbi.GraphicalModel = object

    monkeypatch.setitem(
        sys.modules,
        "mbi",
        fake_mbi,
    )

    class FakeSyntheticDataset:
        def __init__(self, df):
            self.df = df

    class FakeGraphicalModel:
        def __init__(self, encoded_df):
            self.encoded_df = encoded_df

        def synthetic_data(self, rows=None):
            n_rows = len(self.encoded_df) if rows is None else rows

            repeated = pd.concat(
                [self.encoded_df] * ((n_rows // len(self.encoded_df)) + 1),
                ignore_index=True,
            ).iloc[:n_rows]

            return FakeSyntheticDataset(repeated.reset_index(drop=True))

    class FakeAIMMechanism:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(
            self,
            data,
            workload,
            *,
            num_synth_rows=None,
            initial_cliques=None,
        ):
            model = FakeGraphicalModel(data.df)

            synthetic = model.synthetic_data(rows=num_synth_rows)

            return model, synthetic

    import katabatic.models.aim.mechanism as mechanism_module

    monkeypatch.setattr(
        mechanism_module,
        "AIMMechanism",
        FakeAIMMechanism,
    )

    data_dir = tmp_path / "splits"
    synthetic_dir = tmp_path / "synthetic"

    data_dir.mkdir()

    training_df = pd.DataFrame(
        {
            "age": [
                20,
                30,
                40,
                50,
            ],
            "city": [
                "Melbourne",
                "Sydney",
                "Melbourne",
                "Brisbane",
            ],
            "class": [
                "A",
                "B",
                "A",
                "B",
            ],
        }
    )

    training_df.to_csv(
        data_dir / "train_full.csv",
        index=False,
    )

    model = AIMModel(
        max_bins=2,
        max_iters=5,
    )

    result = model.train(
        str(data_dir),
        str(synthetic_dir),
        categorical_cols=[
            "city",
            "class",
        ],
        continuous_cols=[
            "age",
        ],
    )

    assert result is model
    assert model.is_fitted is True

    assert model._resolved_categorical_columns == [
        "city",
        "class",
    ]

    assert model._resolved_continuous_columns == [
        "age",
    ]

    synthetic_df = model.sample(6)

    assert isinstance(
        synthetic_df,
        pd.DataFrame,
    )

    assert synthetic_df.shape == (
        6,
        3,
    )

    assert synthetic_df.columns.tolist() == [
        "age",
        "city",
        "class",
    ]

    assert (synthetic_dir / "x_synth.csv").exists()

    assert (synthetic_dir / "y_synth.csv").exists()

    assert (synthetic_dir / "metadata.json").exists()

    # Remove the mechanism module imported while fake mbi was active.
    # A later integration test must import it again with the real mbi package.
    sys.modules.pop("katabatic.models.aim.mechanism", None)
