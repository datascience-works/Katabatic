from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import tempfile
import threading
import warnings
import weakref
from functools import wraps
from pathlib import Path

from katabatic.artifacts.base import ArtifactConflictError, ArtifactStore

try:
    import fsspec
except ImportError as exc:  # pragma: no cover - exercised via check_dependencies
    fsspec = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_STATE_FILE = ".katabatic-sync-state.json"
_PART_SUFFIX = ".katabatic-part"
# Remote info fields that change when an object is rewritten, across backends.
_VERSION_KEYS = (
    "size",
    "ETag",
    "etag",
    "md5Hash",
    "generation",
    "VersionId",
    "LastModified",
    "last_modified",
    "updated",
    "mtime",
    "created",
)


def _remote_version(info: dict) -> list[str]:
    return [str(info.get(key)) for key in _VERSION_KEYS]


def _hash_stream(fh) -> str:
    digest = hashlib.blake2b(digest_size=20)
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return _hash_stream(fh)


def _key(path: str) -> str:
    """Normalise an artifact path to a relative key, rejecting paths outside the store."""
    key = posixpath.normpath(path.strip("/")) if path.strip("/") else ""
    if key == ".":
        return ""
    if key == ".." or key.startswith("../"):
        raise ValueError(f"Artifact path escapes the store: {path!r}")
    return key


def _under(rel: str, key: str) -> bool:
    return not key or rel == key or rel.startswith(key + "/")


def _locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class FsspecArtifactStore(ArtifactStore):
    """Artifact store on any fsspec filesystem (S3, GCS, Azure, ...) via a local cache.

    ``open_path()`` returns paths inside ``local_cache_dir``, as ``LocalArtifactStore``
    does, so code that reads and writes files directly keeps working. ``sync()``
    uploads files changed locally and ``pull()`` downloads files changed remotely;
    ``save_json()``, ``save_bytes()``, ``load_json()`` and ``exists()`` transfer
    single files themselves.

    Changes are tracked per file with a content hash of the local copy and the
    remote object's version (ETag, generation or timestamp), stored in the cache.

    - Writes never overwrite a remote file this store hasn't seen at its current
      version: ``save_json()``, ``save_bytes()`` and ``sync()`` raise
      ``ArtifactConflictError`` instead. Single-file saves are read back to confirm
      they weren't overwritten concurrently.
    - ``pull()`` keeps unsynced local changes and removes local copies of files
      deleted remotely. Nothing deletes remote files.
    - One instance is thread-safe; don't share ``local_cache_dir`` across processes.

    Known limits: ``sync()`` of the same file from two machines at the same instant
    can still race, and backends that only report timestamps (e.g. local disk) can
    miss a same-size rewrite within one timestamp tick.

    Without ``local_cache_dir``, a temporary cache is used and deleted by ``close()``.
    """

    def __init__(
        self,
        remote_root: str,
        local_cache_dir: str | Path | None = None,
        storage_options: dict | None = None,
    ) -> None:
        self.check_dependencies()
        self.fs, root = fsspec.core.url_to_fs(remote_root, **(storage_options or {}))
        self._remote_root = root.rstrip("/")

        self._finalizer = None
        if local_cache_dir is None:
            local_cache_dir = tempfile.mkdtemp(prefix="katabatic-artifact-cache-")
            self._finalizer = weakref.finalize(
                self, shutil.rmtree, local_cache_dir, ignore_errors=True
            )
        self.root = Path(local_cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self._state_path = self.root / _STATE_FILE
        self._state: dict[str, dict] = (
            json.loads(self._state_path.read_text(encoding="utf-8"))
            if self._state_path.exists()
            else {}
        )
        self._lock = threading.RLock()

    @staticmethod
    def check_dependencies() -> None:
        if fsspec is None:  # pragma: no cover
            raise ImportError(
                "FsspecArtifactStore requires fsspec and your backend's filesystem "
                "package. Install katabatic[artifacts-s3], [artifacts-gcs] or "
                "[artifacts-azure]."
            ) from _IMPORT_ERROR

    def close(self) -> None:
        """Delete the temporary cache, if this store created one."""
        if self._finalizer is not None:
            self._finalizer()

    # -- ArtifactStore contract -----------------------------------------
    @_locked
    def save_json(self, path: str, data: dict) -> None:
        self._write(_key(path), json.dumps(data, indent=2).encode("utf-8"))

    @_locked
    def save_bytes(self, path: str, data: bytes) -> None:
        self._write(_key(path), data)

    @_locked
    def load_json(self, path: str) -> dict:
        key = _key(path)
        self.pull(key)
        return json.loads(self._local(key).read_text(encoding="utf-8"))

    def open_path(self, path: str) -> Path:
        full = self._local(_key(path))
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    @_locked
    def exists(self, path: str) -> bool:
        """Whether ``path`` exists; it is downloaded, so ``open_path()`` can read it."""
        key = _key(path)
        self.pull(key)
        return self._local(key).exists()

    # -- sync/pull ------------------------------------------------------
    @_locked
    def sync(self, path: str = "") -> None:
        """Upload files under ``path`` that changed locally since their last transfer."""
        key = _key(path)
        hashes = {rel: _hash_file(self._local(rel)) for rel in self._local_files(key)}
        changed = [
            rel
            for rel, digest in hashes.items()
            if rel not in self._state or self._state[rel]["hash"] != digest
        ]
        if not changed:
            return

        remote = self._listing(key)
        unseen = [
            rel for rel in changed if rel in remote and not self._seen(rel, remote[rel])
        ]
        if unseen:
            raise ArtifactConflictError(
                f"Not uploading: {len(unseen)} file(s) changed remotely since this store "
                f"last read them, e.g. {unseen[0]}. Pull and resolve them first."
            )

        self._put({rel: self._local(rel) for rel in changed})
        uploaded = self._listing(key)
        for rel in changed:
            self._record(rel, hashes[rel], uploaded[rel])
        self._save_state()

    @_locked
    def pull(self, path: str = "") -> None:
        """Download files under ``path`` that are missing locally or changed remotely."""
        key = _key(path)
        remote = self._listing(key)

        for rel in [r for r in self._state if _under(r, key) and r not in remote]:
            local = self._local(rel)
            if not (local.exists() and self._changed_locally(rel)):
                local.unlink(missing_ok=True)  # deleted remotely
                del self._state[rel]

        to_fetch = {}
        for rel, info in remote.items():
            if self._local(rel).exists():
                if self._seen(rel, info):
                    continue
                if self._changed_locally(rel):
                    warnings.warn(
                        f"Not overwriting {rel}: it has unsynced local changes.",
                        stacklevel=2,
                    )
                    continue
            to_fetch[rel] = info
        self._download(to_fetch)
        self._save_state()

    # -- helpers --------------------------------------------------------
    def _local(self, key: str) -> Path:
        return self.root / key

    def _remote(self, key: str) -> str:
        return f"{self._remote_root}/{key}" if key else self._remote_root

    def _seen(self, rel: str, info: dict) -> bool:
        record = self._state.get(rel)
        return record is not None and record["remote"] == _remote_version(info)

    def _changed_locally(self, rel: str) -> bool:
        record = self._state.get(rel)
        return record is None or _hash_file(self._local(rel)) != record["hash"]

    def _record(self, rel: str, digest: str, info: dict) -> None:
        self._state[rel] = {"hash": digest, "remote": _remote_version(info)}

    def _write(self, key: str, data: bytes) -> None:
        info = self._listing(key).get(key)
        if info is not None and not self._seen(key, info):
            raise ArtifactConflictError(
                f"{key} changed remotely since this store last read it; "
                "load it, re-apply your change, and save again."
            )

        # Upload from a temporary file; the local copy only changes once the
        # remote is confirmed to hold these bytes.
        local = self._local(key)
        part = local.with_name(local.name + _PART_SUFFIX)
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(data)
        digest = _hash_file(part)
        try:
            self._put({key: part})
            info = self._listing(key).get(key)
            with self.fs.open(self._remote(key), "rb") as fh:
                written = _hash_stream(fh)
            if info is None or written != digest:
                raise ArtifactConflictError(
                    f"{key} was overwritten by another writer while saving; "
                    "load it, re-apply your change, and save again."
                )
            os.replace(part, local)
        finally:
            part.unlink(missing_ok=True)
        self._record(key, digest, info)
        self._save_state()

    def _put(self, files: dict[str, Path]) -> None:
        remote_paths = [self._remote(rel) for rel in files]
        for parent in {posixpath.dirname(p) for p in remote_paths}:
            self.fs.makedirs(parent, exist_ok=True)
        self.fs.put([str(p) for p in files.values()], remote_paths)

    def _download(self, infos: dict[str, dict]) -> None:
        if not infos:
            return
        # Download beside the target, then rename, so failures never leave partial files.
        parts = {rel: self._local(rel + _PART_SUFFIX) for rel in infos}
        for part in parts.values():
            part.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get(
            [self._remote(rel) for rel in parts], [str(p) for p in parts.values()]
        )
        for rel, part in parts.items():
            os.replace(part, self._local(rel))
            self._record(rel, _hash_file(self._local(rel)), infos[rel])

    def _listing(self, key: str) -> dict[str, dict]:
        found = self.fs.find(self._remote(key), detail=True)
        start = len(self._remote_root) + 1
        listing = {
            name.split("://", 1)[-1][start:]: info for name, info in found.items()
        }
        # Object stores list by prefix, so "run1" also matches "run10".
        return {rel: info for rel, info in listing.items() if _under(rel, key)}

    def _local_files(self, key: str) -> list[str]:
        base = self._local(key)
        files = (
            [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        )
        return [
            p.relative_to(self.root).as_posix()
            for p in files
            if not p.name.startswith(_STATE_FILE) and not p.name.endswith(_PART_SUFFIX)
        ]

    def _save_state(self) -> None:
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        tmp.write_text(json.dumps(self._state), encoding="utf-8")
        os.replace(tmp, self._state_path)
