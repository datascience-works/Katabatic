from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


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

    # -- Remote-sync extension -------------------------------------------
    #
    # A large amount of existing code (dataset_split.py, the train/test
    # pipeline, and vendored model backends such as CTGAN/GANBLR/PATE-GAN)
    # calls open_path() and then reads/writes files under it directly with
    # pandas, shutil, torch.save, etc. — treating the result as a real,
    # already-populated local folder. That implicit contract can't be
    # satisfied by a naive "point open_path() at a remote URL" swap.
    #
    # Stores that back onto local disk already satisfy this contract for
    # free, so sync()/pull()/workdir() are no-ops here. A remote-backed
    # store (see remote.py) overrides them to stage files locally and
    # push/pull to the remote backend around that same open_path() call.
    def sync(self, path: str = "") -> None:
        """Push any local-only changes under `path` to the backing store.

        No-op for stores that already write directly to their backing
        storage (e.g. LocalArtifactStore). Remote-backed stores override
        this to upload local-cache files to the remote filesystem.
        """
        return None

    def pull(self, path: str = "") -> None:
        """Ensure local content under `path` reflects the backing store.

        No-op for stores that already read directly from their backing
        storage. Remote-backed stores override this to download remote
        objects into the local cache before they're read.
        """
        return None

    @contextmanager
    def workdir(self, path: str) -> Iterator[Path]:
        """Convenience wrapper: pull() before, yield the local Path via
        open_path(), sync() after — even if the block raises.

        Existing call sites don't have to change at all (they can keep
        calling open_path() directly); this is for new/updated call sites
        that want the stage-in/stage-out handled in one place instead of
        two separate pull()/sync() calls.
        """
        self.pull(path)
        try:
            yield self.open_path(path)
        finally:
            self.sync(path)
