from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from katabatic.models.tablegan.models import TableGANModel, _network_side
from katabatic.models.tablegan.utils import (
    build_schema,
    calculate_side_length,
    decode_dataframe,
    encode_dataframe,
    square_to_table,
    table_to_square,
)


@pytest.fixture
def mixed_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "income": [30000.0, 45000.0, 60000.0, 75000.0],
            "city": ["Melbourne", "Sydney", "Melbourne", "Brisbane"],
            "target": [0, 1, 0, 1],
        }
    )


def test_build_schema_with_explicit_columns(
    mixed_dataframe: pd.DataFrame,
) -> None:
    schema = build_schema(
        mixed_dataframe,
        categorical_cols=["city", "target"],
        continuous_cols=["age", "income"],
    )

    kinds = {meta.name: meta.kind for meta in schema}

    assert kinds == {
        "age": "continuous",
        "income": "continuous",
        "city": "categorical",
        "target": "categorical",
    }


def test_encode_dataframe_has_expected_shape(
    mixed_dataframe: pd.DataFrame,
) -> None:
    schema = build_schema(
        mixed_dataframe,
        categorical_cols=["city", "target"],
        continuous_cols=["age", "income"],
    )

    encoded = encode_dataframe(mixed_dataframe, schema)

    assert encoded.shape == mixed_dataframe.shape
    assert encoded.dtype == np.float32
    assert np.all(encoded >= -1.0)
    assert np.all(encoded <= 1.0)


def test_encode_decode_round_trip(
    mixed_dataframe: pd.DataFrame,
) -> None:
    schema = build_schema(
        mixed_dataframe,
        categorical_cols=["city", "target"],
        continuous_cols=["age", "income"],
    )

    encoded = encode_dataframe(mixed_dataframe, schema)
    decoded = decode_dataframe(encoded, schema)

    assert decoded.columns.tolist() == mixed_dataframe.columns.tolist()

    np.testing.assert_allclose(
        decoded["age"].to_numpy(dtype=float),
        mixed_dataframe["age"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        decoded["income"].to_numpy(dtype=float),
        mixed_dataframe["income"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )

    assert decoded["city"].tolist() == mixed_dataframe["city"].tolist()
    assert decoded["target"].tolist() == mixed_dataframe["target"].astype(str).tolist()


def test_calculate_side_length() -> None:
    assert calculate_side_length(1) == 1
    assert calculate_side_length(4) == 2
    assert calculate_side_length(5) == 3
    assert calculate_side_length(9) == 3


def test_network_side() -> None:
    assert _network_side(1) == 4
    assert _network_side(4) == 4
    assert _network_side(16) == 4
    assert _network_side(17) == 8
    assert _network_side(64) == 8
    assert _network_side(65) == 16


def test_table_square_round_trip() -> None:
    encoded = np.array(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0, -0.5, -1.0],
        ],
        dtype=np.float32,
    )

    square = table_to_square(
        encoded,
        side_length=4,
    )

    assert square.shape == (2, 1, 4, 4)

    restored = square_to_table(
        square,
        n_columns=5,
    )

    np.testing.assert_array_equal(restored, encoded)


def test_table_to_square_rejects_small_side() -> None:
    encoded = np.zeros((2, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="too small"):
        table_to_square(
            encoded,
            side_length=2,
        )


def test_sample_before_training_raises() -> None:
    model = TableGANModel(
        epochs=1,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match=r"Call train\(\) before sample"):
        model.sample(5)


def test_evaluate_before_training_raises() -> None:
    model = TableGANModel(
        epochs=1,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match=r"Call train\(\) before evaluate"):
        model.evaluate()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("epochs", 0),
        ("batch_size", 0),
        ("noise_dim", 0),
        ("base_channels", 0),
    ],
)
def test_invalid_constructor_values(
    parameter: str,
    value: int,
) -> None:
    kwargs = {parameter: value}

    with pytest.raises(ValueError):
        TableGANModel(**kwargs)


@pytest.mark.parametrize(
    ("n_columns", "expected_side"),
    [
        (4, 4),
        (16, 4),
        (17, 8),
        (64, 8),
        (65, 16),
    ],
)
def test_generator_discriminator_shapes(
    n_columns: int,
    expected_side: int,
) -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    model = TableGANModel(
        epochs=1,
        batch_size=2,
        noise_dim=16,
        base_channels=8,
        device="cpu",
    )

    model._n_columns = n_columns
    model._side_length = _network_side(n_columns)
    model._device = torch.device("cpu")

    model._build_networks(torch, nn)

    noise = torch.randn(
        2,
        model.cfg["noise_dim"],
        1,
        1,
    )

    with torch.no_grad():
        generated = model.generator(noise)
        scores = model.discriminator(generated)

    assert model._side_length == expected_side
    assert generated.shape == (
        2,
        1,
        expected_side,
        expected_side,
    )
    assert scores.shape == (2,)


def test_information_loss_zero_for_identical_data() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel(
        information_weight=1.0,
        delta_mean=0.0,
        delta_std=0.0,
    )

    data = torch.tensor(
        [
            [[[0.0, 0.5], [1.0, -0.5]]],
            [[[0.2, 0.3], [0.8, -0.2]]],
        ],
        dtype=torch.float32,
    )

    loss = model._information_loss(
        data,
        data.clone(),
        torch,
    )

    assert loss.item() == pytest.approx(0.0)


def test_information_loss_detects_distribution_difference() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel(
        information_weight=1.0,
        delta_mean=0.0,
        delta_std=0.0,
    )

    real = torch.zeros(
        (4, 1, 4, 4),
        dtype=torch.float32,
    )

    fake = torch.ones(
        (4, 1, 4, 4),
        dtype=torch.float32,
    )

    loss = model._information_loss(
        real,
        fake,
        torch,
    )

    assert loss.item() > 0.0


def test_information_loss_respects_threshold() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel(
        information_weight=1.0,
        delta_mean=100.0,
        delta_std=100.0,
    )

    real = torch.zeros(
        (4, 1, 4, 4),
        dtype=torch.float32,
    )

    fake = torch.ones(
        (4, 1, 4, 4),
        dtype=torch.float32,
    )

    loss = model._information_loss(
        real,
        fake,
        torch,
    )

    assert loss.item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("information_weight", -1.0),
        ("delta_mean", -0.1),
        ("delta_std", -0.1),
    ],
)
def test_invalid_information_loss_parameters(
    parameter: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        TableGANModel(**{parameter: value})


def test_discriminator_can_return_features() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    model = TableGANModel(
        epochs=1,
        batch_size=2,
        noise_dim=16,
        base_channels=8,
        device="cpu",
    )

    model._n_columns = 17
    model._side_length = _network_side(17)
    model._device = torch.device("cpu")

    model._build_networks(torch, nn)

    inputs = torch.randn(
        2,
        1,
        model._side_length,
        model._side_length,
    )

    with torch.no_grad():
        probabilities, features = model.discriminator(
            inputs,
            return_features=True,
        )

    assert probabilities.shape == (2,)
    assert features.shape[0] == 2
    assert features.ndim == 4
    assert features.shape[-2:] == (4, 4)


def test_classifier_output_shape() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    model = TableGANModel(
        epochs=1,
        batch_size=2,
        noise_dim=16,
        base_channels=8,
        device="cpu",
    )

    model._n_columns = 17
    model._side_length = _network_side(17)
    model._device = torch.device("cpu")

    model._build_networks(torch, nn)

    inputs = torch.randn(
        2,
        1,
        model._side_length,
        model._side_length,
    )

    with torch.no_grad():
        output = model.classifier(inputs)

    assert output.shape == (2,)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_negative_classifier_weight_rejected() -> None:
    with pytest.raises(ValueError):
        TableGANModel(classifier_weight=-1.0)


def test_classifier_masks_last_table_column() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel()
    model._n_columns = 5
    model._side_length = 4

    matrices = torch.tensor(
        [
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.8, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    classifier_inputs, targets = model._classifier_inputs_and_targets(matrices)

    assert torch.allclose(
        targets,
        torch.tensor([0.8]),
    )

    assert classifier_inputs[0, 0, 1, 0].item() == 0.0

    assert torch.allclose(
        classifier_inputs[0, 0, 0],
        matrices[0, 0, 0],
    )

    # The helper must not modify the original matrix.
    assert torch.isclose(
        matrices[0, 0, 1, 0],
        torch.tensor(0.8),
    )


def test_classifier_mask_uses_correct_position_for_larger_matrix() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel()
    model._n_columns = 17
    model._side_length = _network_side(17)

    matrices = torch.zeros(
        2,
        1,
        model._side_length,
        model._side_length,
    )

    # Column index 16 is the first cell of row 2 in an 8x8 matrix.
    matrices[:, 0, 2, 0] = torch.tensor([0.25, -0.75])

    classifier_inputs, targets = model._classifier_inputs_and_targets(matrices)

    assert torch.allclose(
        targets,
        torch.tensor([0.25, -0.75]),
    )

    assert torch.all(classifier_inputs[:, 0, 2, 0] == 0.0)


def test_classification_loss_is_mean_absolute_error() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel()

    predictions = torch.tensor(
        [0.5, -0.5, 0.0],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [1.0, -1.0, 0.5],
        dtype=torch.float32,
    )

    loss = model._classification_loss(
        predictions,
        targets,
        torch,
    )

    assert torch.isclose(
        loss,
        torch.tensor(0.5),
    )


def test_classifier_mask_requires_initialised_schema() -> None:
    torch = pytest.importorskip("torch")

    model = TableGANModel()

    matrices = torch.zeros(
        1,
        1,
        4,
        4,
    )

    with pytest.raises(
        RuntimeError,
        match="schema has not been initialised",
    ):
        model._classifier_inputs_and_targets(matrices)


def test_artifact_round_trip_restores_all_networks(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef
    from katabatic.models.tablegan.utils import ColumnMeta

    class TestArtifactStore(ArtifactStore):
        def __init__(self, root):
            self.root = root

        def save_json(self, path: str, data: dict) -> None:
            import json

            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)

            with destination.open("w", encoding="utf-8") as file:
                json.dump(data, file)

        def load_json(self, path: str) -> dict:
            import json

            with (self.root / path).open("r", encoding="utf-8") as file:
                return json.load(file)

        def save_bytes(self, path: str, data: bytes) -> None:
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

        def open_path(self, path: str):
            return self.root / path

        def exists(self, path: str) -> bool:
            return (self.root / path).exists()

    model = TableGANModel(
        epochs=1,
        batch_size=2,
        noise_dim=16,
        base_channels=8,
        device="cpu",
    )

    model.columns = [
        "feature_a",
        "feature_b",
        "target",
    ]

    model.schema = [
        ColumnMeta(
            name="feature_a",
            kind="continuous",
            minimum=0.0,
            maximum=1.0,
            original_dtype="float64",
        ),
        ColumnMeta(
            name="feature_b",
            kind="continuous",
            minimum=0.0,
            maximum=1.0,
            original_dtype="float64",
        ),
        ColumnMeta(
            name="target",
            kind="categorical",
            categories=["0", "1"],
            original_dtype="int64",
        ),
    ]

    model._n_columns = len(model.columns)
    model._side_length = _network_side(model._n_columns)
    model._device = torch.device("cpu")
    model.is_fitted = True

    model._build_networks(torch, nn)

    ref = ModelRef(
        model_name="tablegan",
        dataset_name="test-dataset",
        dataset_version="1",
        train_run_id="test-run",
    )

    store = TestArtifactStore(tmp_path)

    state_dir = tmp_path / ref.state_relpath

    model._save_artifact_state(str(state_dir))

    restored = TableGANModel.load_from_ref(
        store,
        ref,
    )

    assert restored.is_fitted is True
    assert restored.columns == model.columns
    assert restored._n_columns == model._n_columns
    assert restored._side_length == model._side_length

    assert restored.generator is not None
    assert restored.discriminator is not None
    assert restored.classifier is not None

    for original, loaded in zip(
        model.generator.parameters(),
        restored.generator.parameters(),
        strict=True,
    ):
        assert torch.equal(original, loaded)

    for original, loaded in zip(
        model.discriminator.parameters(),
        restored.discriminator.parameters(),
        strict=True,
    ):
        assert torch.equal(original, loaded)

    for original, loaded in zip(
        model.classifier.parameters(),
        restored.classifier.parameters(),
        strict=True,
    ):
        assert torch.equal(original, loaded)

    synthetic = restored.sample(3)

    assert synthetic.shape == (3, 3)
    assert synthetic.columns.tolist() == model.columns
    assert set(synthetic["target"]).issubset({"0", "1"})
