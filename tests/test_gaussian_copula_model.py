import pandas as pd

from katabatic.models.gaussian_copula import GaussianCopulaModel
from katabatic.models.gaussian_copula.utils import load_training_dataframe
from katabatic.models.registry import ModelRegistry


def test_gaussian_copula_utils_and_registry(tmp_path):
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        }
    )

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    data.to_csv(train_dir / "train_full.csv", index=False)

    loaded = load_training_dataframe(str(train_dir))
    assert list(loaded.columns) == list(data.columns)
    assert len(loaded) == len(data)

    assert "gaussian_copula" in ModelRegistry.get_available_models()
    assert ModelRegistry.is_supported("gaussian_copula") is False
    assert GaussianCopulaModel.__name__ == "GaussianCopulaModel"
