from __future__ import annotations

import json
import tempfile
from pathlib import Path

from katabatic.artifacts.base import ArtifactStore

try:
    import fsspec
except ImportError as exc:  # pragma: no cover - exercised via check_dependencies
    fsspec = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class FsspecArtifactStore(ArtifactStore):
    """ArtifactStore backed by any fsspec-compatible remote filesystem
    (S3 via s3fs, GCS via gcsfs, Azure via adlfs, etc.), using a local
    staging directory so existing open_path()-based call sites keep
    working unmodified.

    Design: existing code (dataset_split.py, the train/test pipeline, and
    vendored model backends like CTGAN/GANBLR/PATE-GAN) calls open_path()
    and then reads/writes many files under it directly, outside of
    save_json()/save_bytes(). A naive "open_path() returns a remote URI"
    swap breaks all of that. Instead:

    - open_path() always returns a path inside a local staging directory
      (same behaviour as LocalArtifactStore), so every existing call site
      keeps working with zero changes.
    - save_json()/save_bytes() write to the local stage AND immediately
      upload that one file remotely, so tracked/explicit writes are always
      in sync.
    - sync(path) walks the local stage under `path` and uploads anything
      newer than what's remote — call this after bulk writes that bypassed
      save_json/save_bytes (e.g. at the end of write_dataset_artifact(),
      or the model.train()/sample() steps in the pipeline).
    - pull(path) walks the remote store under `path` and downloads
      anything missing or newer locally — call this before reading
      artifacts that may have been written by another process/machine.
    - workdir(path) is pull() + yield local Path + sync(), for callers
      that want both in one call (see ArtifactStore.workdir in base.py).

    Two other options were considered and rejected:
    - Rewriting every call site to work against remote paths directly:
      too invasive, touches every vendored model backend.
    - A FUSE-based mount (s3fs-fuse / gcsfuse) so the remote looks like a
      normal local mount: avoids code changes entirely, but adds an
      external OS-level dependency/mount step outside Python's control,
      and network-latency-per-syscall behaviour is a poor fit for the
      many small file writes model training does.
    """

    def __init__(
        self,
        remote_root: str,
        local_cache_dir: str | Path | None = None,
        storage_options: dict | None = None,
    ) -> None:
        self.check_dependencies()

        self.fs, self._remote_root = fsspec.core.url_to_fs(
            remote_root, **(storage_options or {})
        )
        self._remote_root = self._remote_root.rstrip("/")

        if local_cache_dir is None:
            local_cache_dir = tempfile.mkdtemp(prefix="katabatic-artifact-cache-")
        self.root = Path(local_cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        # Tracks, per relpath, the local mtime we last successfully
        # transferred (either direction). Lets sync()/pull() skip files
        # that haven't changed locally since we last touched them, without
        # relying on remote size/mtime (which a size-only check gets
        # wrong for same-size edits, and which not every fsspec backend
        # reports at matching precision anyway).
        self._last_synced_mtime: dict[str, float] = {}

    @staticmethod
    def check_dependencies() -> None:
        if fsspec is None:  # pragma: no cover
            raise ImportError(
                "FsspecArtifactStore requires the 'fsspec' package (and a "
                "filesystem-specific package such as s3fs, gcsfs, or adlfs "
                "for your backend). Install via "
                "pip install katabatic[artifacts-remote]."
            ) from _IMPORT_ERROR

    # -- path helpers ------------------------------------------------
    def _local(self, path: str) -> Path:
        return self.root / path

    def _remote(self, path: str) -> str:
        path = path.strip("/")
        return f"{self._remote_root}/{path}" if path else self._remote_root

    # -- ArtifactStore contract ---------------------------------------
    def save_json(self, path: str, data: dict) -> None:
        full = self._local(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._upload_file(path)

    def load_json(self, path: str) -> dict:
        if not self._local(path).exists():
            self._download_file(path)
        return json.loads(self._local(path).read_text(encoding="utf-8"))

    def save_bytes(self, path: str, data: bytes) -> None:
        full = self._local(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        self._upload_file(path)

    def open_path(self, path: str) -> Path:
        full = self._local(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def exists(self, path: str) -> bool:
        if self._local(path).exists():
            return True
        return self.fs.exists(self._remote(path))

    # -- sync/pull ------------------------------------------------------
    def sync(self, path: str = "") -> None:
        local_root = self._local(path)
        if not local_root.exists():
            return
        if local_root.is_file():
            self._upload_file(path)
            return
        for local_file in local_root.rglob("*"):
            if local_file.is_file():
                rel = local_file.relative_to(self.root).as_posix()
                self._upload_file(rel)

    def pull(self, path: str = "") -> None:
        remote_path = self._remote(path)
        if not self.fs.exists(remote_path):
            return
        info = self.fs.info(remote_path)
        if info["type"] == "file":
            self._download_file(path)
            return
        for remote_file in self.fs.find(remote_path):
            rel = remote_file[len(self._remote_root) :].lstrip("/")
            self._download_file(rel)

    # -- single-file transfer, skipping unchanged files ------------------
    def _upload_file(self, path: str) -> None:
        local_file = self._local(path)
        remote_file = self._remote(path)
        local_mtime = local_file.stat().st_mtime
        if self._last_synced_mtime.get(path) == local_mtime and self.fs.exists(
            remote_file
        ):
            return  # local file unchanged since we last uploaded it
        self.fs.makedirs(self._parent(remote_file), exist_ok=True)
        self.fs.put_file(str(local_file), remote_file)
        self._last_synced_mtime[path] = local_mtime

    def _download_file(self, path: str) -> None:
        local_file = self._local(path)
        remote_file = self._remote(path)
        if (
            local_file.exists()
            and self._last_synced_mtime.get(path) == local_file.stat().st_mtime
        ):
            return  # local file already matches what we last downloaded
        local_file.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get_file(remote_file, str(local_file))
        self._last_synced_mtime[path] = local_file.stat().st_mtime

    def _parent(self, remote_path: str) -> str:
        return remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
