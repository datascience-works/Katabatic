"""Every GReaT benchmark script must avoid the known causes of timeouts."""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "benchmarks" / "examples" / "great"
SCRIPTS = sorted(EXAMPLES.glob("run_great_*.py"))


def test_all_five_datasets_have_a_script():
    assert [p.stem.removeprefix("run_great_") for p in SCRIPTS] == [
        "adult",
        "car",
        "magic",
        "nursery",
        "shuttle",
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_each_dataset_has_its_own_checkpoint_folder(script):
    dataset = script.stem.removeprefix("run_great_")

    assert f'EXPERIMENT_DIR = "trainer_great_{dataset}"' in script.read_text()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_script_does_not_call_train(script):
    text = script.read_text()

    # train() samples every row once more on the CPU with a tiny batch.
    assert "model.train(" not in text
    assert "model.fit(train_df)" in text


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_sampling_goes_through_the_chunked_helper(script):
    text = script.read_text()

    assert "from great_sampling import sample_in_chunks" in text
    assert "model.sample(len(train_df)" not in text
    # Stability evaluation resamples the model, so evaluate() must get the
    # chunked sampler too, not the raw model whose default device is the CPU.
    assert "SimpleNamespace(sample=sample_for_evaluation)" in text


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_training_and_sampling_are_timed_as_separate_phases(script):
    text = script.read_text()

    assert 'with runtime.phase("train"):' in text
    assert 'with runtime.phase("sample"):' in text
    # Stability resampling also goes through sample_for_evaluation.
    assert 'with runtime.phase("sample_total"):' in text
