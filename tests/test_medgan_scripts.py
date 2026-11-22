"""Every MedGAN benchmark script must encode categories before training."""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "benchmarks" / "examples" / "medgan"
SCRIPTS = sorted(EXAMPLES.glob("run_medgan_*.py"))


def test_all_five_datasets_have_a_script():
    assert [p.stem.removeprefix("run_medgan_") for p in SCRIPTS] == [
        "adult",
        "car",
        "magic",
        "nursery",
        "shuttle",
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_script_decodes_samples_before_saving(script):
    text = script.read_text()

    # The model returns a numpy array; save_synthetic needs a DataFrame in the
    # original schema, so every script must decode through sample_dataframe.
    assert "def sample_dataframe" in text
    assert "save_synthetic(\n    sample_dataframe(" in text
    assert "model.sample(len(train_df))" not in text


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_script_never_trains_on_the_raw_split_directory(script):
    text = script.read_text()

    if script.stem.endswith("shuttle"):
        pytest.skip("Shuttle is numeric and trains on the split directory")
    assert 'model.train(\n    paths["split_dir"]' not in text
