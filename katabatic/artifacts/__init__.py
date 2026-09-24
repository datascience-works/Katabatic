from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.dataset_split import (
    write_dataset_artifact,
    write_dataset_artifact_presplit,
)
from katabatic.artifacts.ids import new_eval_id, new_split_id, new_train_id
from katabatic.artifacts.local import LocalArtifactStore
from katabatic.artifacts.refs import (
    DatasetRef,
    EvaluationRef,
    ModelRef,
    artifact_path_segment,
)

try:
    from katabatic.artifacts.remote import FsspecArtifactStore
except ImportError:
    FsspecArtifactStore = None  # fsspec extra not installed

__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "FsspecArtifactStore",
    "DatasetRef",
    "ModelRef",
    "EvaluationRef",
    "artifact_path_segment",
    "write_dataset_artifact",
    "write_dataset_artifact_presplit",
    "new_split_id",
    "new_train_id",
    "new_eval_id",
]
