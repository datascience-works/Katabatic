"""Tests for FsspecArtifactStore, using fsspec's in-memory backend (no cloud credentials)."""

from __future__ import annotations

import gc
import json
import threading
import uuid

import pytest

from tests.conftest import require_backend

require_backend("fsspec", "core")

from katabatic.artifacts import ArtifactConflictError  # noqa: E402
from katabatic.artifacts.remote import FsspecArtifactStore  # noqa: E402
from katabatic.datasets.registry import DatasetRegistry  # noqa: E402

pytestmark = pytest.mark.artifacts_remote


@pytest.fixture
def remote_root():
    # memory:// is process-global, so give each test its own namespace.
    return f"memory://katabatic-test-{uuid.uuid4().hex}"


@pytest.fixture
def make_store(remote_root, tmp_path):
    """Stores sharing one remote, each with its own cache: one per 'machine'."""

    def make(name="cache"):
        return FsspecArtifactStore(remote_root, local_cache_dir=tmp_path / name)

    return make


@pytest.fixture
def store(make_store):
    return make_store()


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,y\n1,0\n2,1\n")
    return str(path)


def _write(store, rel, data: bytes):
    path = store.open_path(rel)
    path.write_bytes(data)
    return path


# -- transfer ---------------------------------------------------------------


def test_save_json_is_readable_from_another_machine(store, make_store):
    store.save_json("m/config.json", {"a": 1})
    assert make_store("other").load_json("m/config.json") == {"a": 1}


def test_sync_then_pull_moves_files_written_directly(store, make_store):
    _write(store, "d/train/train.csv", b"a,b\n1,2\n")
    store.sync("d")

    other = make_store("other")
    other.pull("d")
    assert other.open_path("d/train/train.csv").read_bytes() == b"a,b\n1,2\n"


def test_exists_checks_the_remote(store, make_store):
    store.save_json("m/config.json", {})
    other = make_store("other")
    assert other.exists("m/config.json")
    assert not other.exists("m/missing.json")


def test_workdir_pulls_before_and_syncs_after(store, make_store):
    _write(store, "m/state/w.bin", b"v1")
    store.sync("m")

    with make_store("other").workdir("m/state") as local_dir:
        assert (local_dir / "w.bin").read_bytes() == b"v1"
        (local_dir / "w.bin").write_bytes(b"v2")

    check = make_store("check")
    check.pull("m")
    assert check.open_path("m/state/w.bin").read_bytes() == b"v2"


def test_workdir_uploads_nothing_when_the_block_raises(store, make_store):
    def crashing_run():
        with store.workdir("m/state") as local_dir:
            (local_dir / "partial.bin").write_bytes(b"half")
            raise RuntimeError("training crashed")

    with pytest.raises(RuntimeError):
        crashing_run()
    assert not make_store("other").exists("m/state/partial.bin")


def test_sync_uploads_changed_files_in_one_batch(store, monkeypatch):
    for i in range(20):
        _write(store, f"m/part{i}.bin", bytes([i]))

    real_put = store.fs.put
    batches = []
    monkeypatch.setattr(
        store.fs, "put", lambda lp, rp, **k: (batches.append(len(lp)), real_put(lp, rp))
    )
    store.sync("m")
    store.sync("m")  # nothing changed
    assert batches == [20]


def test_sync_record_survives_a_restart(store, make_store, monkeypatch):
    store.save_bytes("m/w.bin", b"v1")

    restarted = make_store()  # same cache dir
    calls = []
    monkeypatch.setattr(restarted.fs, "put", lambda *a, **k: calls.append("put"))
    monkeypatch.setattr(restarted.fs, "get", lambda *a, **k: calls.append("get"))
    restarted.sync("m")
    restarted.pull("m")
    assert calls == []


def test_listing_does_not_match_sibling_prefixes(store, make_store):
    store.save_json("m/run1/a.json", {})
    store.save_json("m/run10/a.json", {})

    other = make_store("other")
    other.pull("m/run1")
    assert other.open_path("m/run1/a.json").exists()
    assert not other.open_path("m/run10/a.json").exists()


# -- change detection and conflicts -----------------------------------------


def test_rapid_rewrites_all_reach_the_remote(store, make_store):
    reader = make_store("reader")
    for i in range(100):
        store.save_json("x.json", {"i": i})
        assert reader.load_json("x.json") == {"i": i}


def test_pull_fetches_files_changed_remotely(store, make_store):
    reader = make_store("reader")
    store.save_json("x.json", {"v": 1})
    reader.pull("x.json")
    store.save_json("x.json", {"v": 2})
    assert reader.load_json("x.json") == {"v": 2}


def test_pull_keeps_unsynced_local_changes(store, make_store):
    store.save_bytes("m/w.bin", b"v1")
    local = _write(store, "m/w.bin", b"local edit")
    other = make_store("other")
    other.pull("m")
    other.save_bytes("m/w.bin", b"remote edit")

    with pytest.warns(UserWarning, match="unsynced local changes"):
        store.pull("m")
    assert local.read_bytes() == b"local edit"


def test_sync_leaves_a_newer_remote_copy_alone(store, make_store):
    store.save_bytes("m/w.bin", b"v1")
    other = make_store("other")
    other.pull("m")
    other.save_bytes("m/w.bin", b"v2")

    store.sync("m")
    check = make_store("check")
    check.pull("m")
    assert check.open_path("m/w.bin").read_bytes() == b"v2"


def test_save_raises_instead_of_overwriting_a_newer_remote(store, make_store):
    store.save_json("x.json", {"v": 1})
    other = make_store("other")
    other.load_json("x.json")
    other.save_json("x.json", {"v": 2})

    with pytest.raises(ArtifactConflictError):
        store.save_json("x.json", {"v": 3})

    store.load_json("x.json")  # reload, then the save goes through
    store.save_json("x.json", {"v": 3})
    assert make_store("check").load_json("x.json") == {"v": 3}


# -- deletions --------------------------------------------------------------


def test_pull_removes_unchanged_copies_of_remote_deletions(store, make_store):
    store.save_json("m/kept.json", {})
    store.save_json("m/deleted.json", {})
    store.save_json("m/edited.json", {})
    other = make_store("other")
    other.pull("m")
    _write(other, "m/edited.json", b'{"edited": true}')

    store.fs.rm(store._remote("m/deleted.json"))
    store.fs.rm(store._remote("m/edited.json"))
    other.pull("m")

    assert other.open_path("m/kept.json").exists()
    assert not other.open_path("m/deleted.json").exists()
    assert other.open_path("m/edited.json").exists()  # unsynced changes are kept


def test_sync_does_not_resurrect_remote_deletions(store, make_store):
    store.save_json("m/x.json", {})
    store.fs.rm(store._remote("m/x.json"))

    store.sync("m")
    assert not make_store("other").exists("m/x.json")


def test_local_deletions_are_not_propagated(store, make_store):
    store.save_json("m/x.json", {"v": 1})
    store.open_path("m/x.json").unlink()
    store.sync("m")
    assert make_store("other").load_json("m/x.json") == {"v": 1}


# -- registry, lifecycle, threads -------------------------------------------


def test_registry_retries_when_another_worker_saves_in_between(
    make_store, csv_path, monkeypatch
):
    worker_a, worker_b = make_store("a"), make_store("b")
    DatasetRegistry(worker_a).register_if_absent("magic", csv_path, target_column="y")

    real_save = worker_a.save_json

    def save_after_b(path, data):
        monkeypatch.setattr(worker_a, "save_json", real_save)
        DatasetRegistry(worker_b).register_if_absent("car", csv_path, target_column="y")
        real_save(path, data)

    monkeypatch.setattr(worker_a, "save_json", save_after_b)
    DatasetRegistry(worker_a).register_if_absent("adult", csv_path, target_column="y")

    registry = json.dumps(make_store("check").load_json("registry/datasets.json"))
    assert all(name in registry for name in ("magic", "car", "adult"))


def test_register_if_absent_returns_a_concurrently_added_entry(
    make_store, csv_path, monkeypatch
):
    worker_a, worker_b = make_store("a"), make_store("b")
    worker_a.save_json("registry/datasets.json", {"datasets": {}})

    real_save = worker_a.save_json

    def b_registers_first(path, data):
        monkeypatch.setattr(worker_a, "save_json", real_save)
        DatasetRegistry(worker_b).register_if_absent("car", csv_path, target_column="y")
        real_save(path, data)

    monkeypatch.setattr(worker_a, "save_json", b_registers_first)
    entry = DatasetRegistry(worker_a).register_if_absent(
        "car", csv_path, target_column="y"
    )
    assert entry["target_column"] == "y"


def test_close_deletes_only_a_temporary_cache(store, remote_root):
    temporary = FsspecArtifactStore(remote_root)
    temporary.save_json("x.json", {})
    temporary.close()
    assert not temporary.root.exists()

    unreferenced = FsspecArtifactStore(remote_root)
    cache = unreferenced.root
    del unreferenced
    gc.collect()
    assert not cache.exists()

    store.save_json("y.json", {})
    store.close()
    assert store.root.exists()


def test_one_store_can_be_shared_across_threads(store, make_store):
    def write(i):
        for j in range(10):
            store.save_json(f"t/{i}.json", {"j": j})

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check = make_store("check")
    assert all(check.load_json(f"t/{i}.json") == {"j": 9} for i in range(8))


def test_model_trained_on_one_machine_reloads_on_another(make_store, tiny_binary_csv):
    from katabatic.models.naivebayes.models import NaiveBayesModel
    from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

    res = TrainTestSplitPipeline(model=NaiveBayesModel()).run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=make_store("trainer"),
        model_name="naivebayes",
        test_size=0.3,
        seed=42,
    )

    reloaded = NaiveBayesModel.load_from_ref(
        make_store("other-machine"), res["model_ref"]
    )
    assert len(reloaded.sample(5)) == 5


# -- conflicts on first write and when both sides change ---------------------


def test_first_save_does_not_overwrite_an_unseen_remote_file(store, make_store):
    make_store("other").save_json("x.json", {"by": "other"})

    with pytest.raises(ArtifactConflictError):
        store.save_json("x.json", {"by": "store"})
    assert make_store("check").load_json("x.json") == {"by": "other"}


def test_fresh_workers_creating_the_registry_keep_both_entries(
    make_store, csv_path, monkeypatch
):
    worker_a, worker_b = make_store("a"), make_store("b")

    real_save = worker_a.save_json

    def b_creates_it_first(path, data):
        monkeypatch.setattr(worker_a, "save_json", real_save)
        DatasetRegistry(worker_b).register_if_absent("car", csv_path, target_column="y")
        real_save(path, data)

    monkeypatch.setattr(worker_a, "save_json", b_creates_it_first)
    DatasetRegistry(worker_a).register_if_absent("adult", csv_path, target_column="y")

    registry = json.dumps(make_store("check").load_json("registry/datasets.json"))
    assert "car" in registry and "adult" in registry


def test_sync_refuses_to_overwrite_when_both_sides_changed(store, make_store):
    store.save_bytes("m/w.bin", b"v1")
    other = make_store("other")
    other.pull("m")
    other.save_bytes("m/w.bin", b"v2 remote")
    _write(store, "m/w.bin", b"v2 local")

    with pytest.raises(ArtifactConflictError):
        store.sync("m")
    check = make_store("check")
    check.pull("m")
    assert check.open_path("m/w.bin").read_bytes() == b"v2 remote"


# -- failures and races during a save ----------------------------------------


def test_failed_upload_leaves_the_store_usable(store, make_store, monkeypatch):
    store.save_json("x.json", {"v": 1})
    other = make_store("other")
    other.load_json("x.json")

    def failing_put(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(store.fs, "put", failing_put)
    with pytest.raises(OSError):
        store.save_json("x.json", {"v": "lost"})
    monkeypatch.undo()

    other.save_json("x.json", {"v": 2})
    assert store.load_json("x.json") == {"v": 2}  # no stale local copy
    store.save_json("x.json", {"v": 3})
    assert other.load_json("x.json") == {"v": 3}


def test_save_detects_a_concurrent_overwrite(store, make_store, monkeypatch):
    store.save_json("x.json", {"v": 1})
    real_put = store.fs.put

    def put_then_race(lpaths, rpaths, **kwargs):
        real_put(lpaths, rpaths, **kwargs)
        store.fs.pipe_file(rpaths[0], b'{"v": "racer"}')

    monkeypatch.setattr(store.fs, "put", put_then_race)
    with pytest.raises(ArtifactConflictError):
        store.save_json("x.json", {"v": 2})
    monkeypatch.undo()

    assert store.load_json("x.json") == {"v": "racer"}


# -- reads -------------------------------------------------------------------


def test_exists_makes_remote_files_readable_through_open_path(store, make_store):
    store.save_json("m/x.json", {"v": 1})
    other = make_store("other")
    assert other.exists("m/x.json")
    assert json.loads(other.open_path("m/x.json").read_text()) == {"v": 1}

    store.fs.rm(store._remote("m/x.json"))
    assert not other.exists("m/x.json")


def test_paths_are_normalised_and_cannot_escape_the_store(store, make_store):
    store.save_json("/lead/x.json", {"v": 1})
    assert store.open_path("lead/x.json").is_relative_to(store.root)
    assert make_store("other").load_json("lead/./x.json") == {"v": 1}

    for bad in ("../outside.json", "a/../../outside.json"):
        with pytest.raises(ValueError, match="escapes the store"):
            store.save_json(bad, {})


def test_state_helpers_download_the_whole_state_directory(store, make_store):
    from katabatic.artifacts.refs import ModelRef
    from katabatic.models.base_model import Model

    class _Model(Model):
        ARTIFACT_STATE_FILES = ("state.json",)

        def train(self, *args, **kwargs):
            return self

        def evaluate(self, *args, **kwargs):
            return {}

        def sample(self, *args, **kwargs):
            return None

    ref = ModelRef(
        model_name="m", dataset_name="d", dataset_version="v1", train_run_id="r1"
    )
    store.save_json(f"{ref.state_relpath}/state.json", {})
    _write(store, f"{ref.state_relpath}/weights/model.bin", b"weights")
    store.sync(ref.state_relpath)

    other = make_store("other")
    state_file = _Model._require_state_file(other, ref)
    assert (state_file.parent / "weights" / "model.bin").read_bytes() == b"weights"


def test_standalone_evaluation_uploads_its_report(make_store, tiny_binary_csv):
    from katabatic.evaluate.fidelity.evaluation import FidelityEvaluation
    from katabatic.models.naivebayes.models import NaiveBayesModel
    from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

    res = TrainTestSplitPipeline(model=NaiveBayesModel()).run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=make_store("trainer"),
        model_name="naivebayes",
        test_size=0.3,
        seed=42,
    )

    evaluation, eval_ref = FidelityEvaluation.from_artifact(
        make_store("evaluator"), res["model_ref"], res["dataset_ref"]
    )
    evaluation.evaluate()

    assert make_store("check").exists(eval_ref.report_relpath)
