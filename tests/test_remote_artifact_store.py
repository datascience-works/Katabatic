"""Tests for FsspecArtifactStore, using fsspec's built-in memory:// backend
so no real S3/GCS/Azure credentials are needed in CI. The same store class
is used against real backends in prod by passing e.g. remote_root="s3://...".
"""
from __future__ import annotations

import uuid

import pytest

from tests.conftest import require_backend

require_backend("fsspec", "core")

from katabatic.artifacts.remote import FsspecArtifactStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    # A fresh memory:// namespace per test so tests don't see each other's
    # files (fsspec's MemoryFileSystem is process-global).
    remote_root = f"memory://katabatic-test-{uuid.uuid4().hex}"
    return FsspecArtifactStore(remote_root, local_cache_dir=tmp_path / "cache")


@pytest.mark.artifacts_remote
def test_save_json_uploads_immediately(store):
    store.save_json("models/m1/config.json", {"a": 1})

    # A second store with an empty local cache, same remote root, can read
    # it back — proves it actually reached the remote backend, not just
    # the local cache.
    other = FsspecArtifactStore(
        f"memory://{store._remote_root.removeprefix('memory://')}",
        local_cache_dir=store.root.parent / "cache2",
    )
    assert other.load_json("models/m1/config.json") == {"a": 1}


@pytest.mark.artifacts_remote
def test_open_path_matches_local_store_contract(store):
    # This is the behaviour every existing call site (dataset_split.py,
    # the pipeline, CTGAN/GANBLR/PATE-GAN) relies on: open_path() returns
    # a real, writable local Path, parent dirs already created.
    p = store.open_path("datasets/adult/split-x/train")
    assert p.parent.exists()
    p.mkdir(parents=True, exist_ok=True)
    (p / "train_full.csv").write_text("a,b\n1,2\n")
    assert (p / "train_full.csv").exists()


@pytest.mark.artifacts_remote
def test_sync_pushes_files_written_outside_save_json(store):
    # Simulates dataset_split.py / model backends: write directly under
    # open_path() (pandas.to_csv, shutil.copy2, torch.save, ...) rather
    # than through save_json/save_bytes, then sync() explicitly.
    p = store.open_path("datasets/adult/split-x/train")
    p.mkdir(parents=True, exist_ok=True)
    (p / "train_full.csv").write_text("a,b\n1,2\n")

    store.sync("datasets/adult/split-x")

    remote_file = store._remote("datasets/adult/split-x/train/train_full.csv")
    assert store.fs.exists(remote_file)


@pytest.mark.artifacts_remote
def test_pull_downloads_into_a_fresh_local_cache(store, tmp_path):
    p = store.open_path("datasets/adult/split-x/train")
    p.mkdir(parents=True, exist_ok=True)
    (p / "train_full.csv").write_text("a,b\n1,2\n")
    store.sync("datasets/adult/split-x")

    # A second store pointed at the same remote root but a brand new,
    # empty local cache -- as if a different machine picked up the job.
    fresh = FsspecArtifactStore(
        f"memory://{store._remote_root.removeprefix('memory://')}",
        local_cache_dir=tmp_path / "fresh-cache",
    )
    assert not fresh.open_path("datasets/adult/split-x/train/train_full.csv").exists()

    fresh.pull("datasets/adult/split-x")

    local_copy = fresh.open_path("datasets/adult/split-x/train/train_full.csv")
    assert local_copy.exists()
    assert local_copy.read_text() == "a,b\n1,2\n"


@pytest.mark.artifacts_remote
def test_workdir_pulls_then_syncs_automatically(store, tmp_path):
    # First "run": write and sync.
    p = store.open_path("models/m1/state")
    p.mkdir(parents=True, exist_ok=True)
    (p / "weights.bin").write_bytes(b"v1")
    store.sync("models/m1/state")

    # Second "run", fresh local cache, uses workdir() instead of manual
    # pull()/sync() -- should see v1 on entry, and any changes should
    # land back on the remote on exit.
    fresh = FsspecArtifactStore(
        f"memory://{store._remote_root.removeprefix('memory://')}",
        local_cache_dir=tmp_path / "fresh-cache",
    )
    with fresh.workdir("models/m1/state") as local_dir:
        assert (local_dir / "weights.bin").read_bytes() == b"v1"
        (local_dir / "weights.bin").write_bytes(b"v2")

    again = FsspecArtifactStore(
        f"memory://{store._remote_root.removeprefix('memory://')}",
        local_cache_dir=tmp_path / "third-cache",
    )
    again.pull("models/m1/state")
    assert (
        again.open_path("models/m1/state/weights.bin").read_bytes() == b"v2"
    )


@pytest.mark.artifacts_remote
def test_exists_checks_remote_when_not_cached_locally(store, tmp_path):
    store.save_json("models/m1/config.json", {"a": 1})

    fresh = FsspecArtifactStore(
        f"memory://{store._remote_root.removeprefix('memory://')}",
        local_cache_dir=tmp_path / "fresh-cache",
    )
    assert fresh.exists("models/m1/config.json")
    assert not fresh.exists("models/m1/does-not-exist.json")


@pytest.mark.artifacts_remote
def test_sync_is_a_noop_for_unchanged_files(store):
    p = store.open_path("models/m1/state")
    p.mkdir(parents=True, exist_ok=True)
    (p / "weights.bin").write_bytes(b"v1")
    store.sync("models/m1/state")

    remote_file = store._remote("models/m1/state/weights.bin")
    before = store.fs.info(remote_file)["size"]

    # Re-sync without changing anything: same-size skip-check means this
    # shouldn't error and the remote copy should be unaffected.
    store.sync("models/m1/state")
    after = store.fs.info(remote_file)["size"]
    assert before == after
