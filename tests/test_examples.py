"""Run every notebook in examples/ so the documented workflows can't go stale."""

from __future__ import annotations

from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.ipynb"))


def test_examples_exist():
    assert EXAMPLES, "no notebooks found in examples/"


@pytest.mark.parametrize("notebook", EXAMPLES, ids=lambda path: path.stem)
def test_example_notebook_runs(notebook, tmp_path):
    nb = nbformat.read(notebook, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        kernel_name="python3",
        timeout=600,
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
