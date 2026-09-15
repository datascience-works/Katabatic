from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.ids import new_split_id
from katabatic.artifacts.refs import DatasetRef
from katabatic.utils.split_dataset import compute_train_test_split
from katabatic.utils.train_test_consistency import sanity_check_train_test


def _copy_extra_assets(
    source_dir: Path | None,
    extra_dir: Path,
) -> None:
    """Copy info.json and TabSyn-style *.npy from source_dir into extra_dir if present."""
    if source_dir is None or not source_dir.is_dir():
        return
    extra_dir.mkdir(parents=True, exist_ok=True)
    info = source_dir / "info.json"
    if info.exists():
        shutil.copy2(info, extra_dir / "info.json")
    for name in (
        "X_num_train.npy",
        "X_cat_train.npy",
        "y_train.npy",
        "X_num_test.npy",
        "X_cat_test.npy",
        "y_test.npy",
    ):
        p = source_dir / name
        if p.exists():
            shutil.copy2(p, extra_dir / name)


def _persist_frames_to_store(
    store: ArtifactStore,
    ref: DatasetRef,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    extra_source_dir: Path | None,
) -> tuple[pd.Series, pd.Series]:
    label_name = df_train.columns[-1]
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
    y_train.name = label_name
    y_test.name = label_name

    train_dir = store.open_path(ref.train_relpath)
    test_dir = store.open_path(ref.test_relpath)
    extra_dir = store.open_path(ref.extra_relpath)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    extra_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(train_dir / "train_full.csv", index=False)
    df_test.to_csv(test_dir / "test_full.csv", index=False)
    X_train.to_csv(train_dir / "x_train.csv", index=False)
    y_train.to_csv(train_dir / "y_train.csv", index=False, header=True)
    X_test.to_csv(test_dir / "x_test.csv", index=False)
    y_test.to_csv(test_dir / "y_test.csv", index=False, header=True)

    shutil.copy2(train_dir / "x_train.csv", extra_dir / "x_train.csv")
    shutil.copy2(train_dir / "y_train.csv", extra_dir / "y_train.csv")
    shutil.copy2(test_dir / "x_test.csv", extra_dir / "x_test.csv")
    shutil.copy2(test_dir / "y_test.csv", extra_dir / "y_test.csv")

    _copy_extra_assets(extra_source_dir, extra_dir)
    return y_train, y_test


def write_dataset_artifact(
    store: ArtifactStore,
    *,
    input_csv: str | Path,
    dataset_name: str,
    test_size: float = 0.2,
    seed: int = 42,
    dataset_version: str | None = None,
    extra_source_dir: str | Path | None = None,
) -> DatasetRef:
    """
    Split input_csv into train/ and test/ under datasets/<name>/<version>/,
    optional extra/ from extra_source_dir (e.g. TabSyn npy + info.json).
    """
    input_csv = Path(input_csv)
    version = dataset_version or new_split_id()
    ref = DatasetRef(dataset_name=dataset_name, dataset_version=version)

    df = pd.read_csv(input_csv)
    print(f"Loaded data with shape: {df.shape}")

    df_train, df_test, _, _, _, _ = compute_train_test_split(
        df, test_size=test_size, seed=seed
    )

    src = Path(extra_source_dir) if extra_source_dir else input_csv.parent
    y_train, y_test = _persist_frames_to_store(store, ref, df_train, df_test, src)

    print("Train label distribution:\n", y_train.value_counts(normalize=True))
    print("Test label distribution:\n", y_test.value_counts(normalize=True))
    print(f"Saved dataset artifact under {ref.root_relpath}")
    return ref


def write_dataset_artifact_presplit(
    store: ArtifactStore,
    *,
    train_csv: str | Path,
    test_csv: str | Path,
    dataset_name: str,
    dataset_version: str | None = None,
    extra_source_dir: str | Path | None = None,
) -> DatasetRef:
    """
    Write user-provided train/test CSVs under datasets/<name>/<version>/
    without performing a random split.
    """
    train_csv = Path(train_csv)
    test_csv = Path(test_csv)
    version = dataset_version or new_split_id()
    ref = DatasetRef(dataset_name=dataset_name, dataset_version=version)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    print(f"Loaded presplit train {df_train.shape}, test {df_test.shape}")
    sanity_check_train_test(df_train, df_test)

    src = Path(extra_source_dir) if extra_source_dir else train_csv.parent
    y_train, y_test = _persist_frames_to_store(store, ref, df_train, df_test, src)

    print("Train label distribution:\n", y_train.value_counts(normalize=True))
    print("Test label distribution:\n", y_test.value_counts(normalize=True))
    print(f"Saved dataset artifact under {ref.root_relpath}")
    return ref
