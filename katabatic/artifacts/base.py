from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ArtifactConflictError(RuntimeError):
    """An artifact changed in the backing store since this store last read or wrote it."""


class ArtifactStore(ABC):
    @abstractmethod
    def save_json(self, path: str, data: dict) -> None: ...

    @abstractmethod
    def load_json(self, path: str) -> dict: ...

    @abstractmethod
    def save_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def open_path(self, path: str) -> Path: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    # Remote-backed stores stage files locally; these hooks keep the stage and the
    # remote in step. They are no-ops for stores that write to disk directly.
    def sync(self, path: str = "") -> None:
        """Upload local changes under ``path`` to the backing store."""

    def pull(self, path: str = "") -> None:
        """Download changes under ``path`` from the backing store."""

    @contextmanager
    def workdir(self, path: str) -> Iterator[Path]:
        """Yield the local directory for ``path``: pull() before, sync() after success."""
        self.pull(path)
        local = self.open_path(path)
        local.mkdir(parents=True, exist_ok=True)
        yield local
        self.sync(path)
