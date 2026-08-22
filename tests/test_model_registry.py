import importlib
import os

import pytest

from katabatic.models.registry import ModelRegistry


def get_supported_models():
    """Extract (name, config) for every model marked supported in the registry."""
    supported = []
    for model_name in ModelRegistry.get_supported_models():
        config = ModelRegistry.get_model_config(model_name)
        supported.append((model_name, config))
    if not supported:
        raise ValueError("No supported models found in ModelRegistry.")
    return supported


MODELS_TO_TEST = get_supported_models()


@pytest.mark.parametrize("model_name, config", MODELS_TO_TEST)
def test_model_promotion_contract(model_name, config):
    """Contract test: every supported model must meet the promotion contract."""

    # 1. Registry entry has a non-null extra and matches the model name.
    assert config.get("extra"), f"'{model_name}' missing/empty 'extra' in registry."
    assert config["extra"] == model_name, (
        f"'{model_name}' extra is '{config['extra']}'. Convention is extra == model name."
    )

    # 2. Module and class are declared
    module_path = config.get("module")
    class_name = config.get("class")
    assert module_path is not None, f"'{model_name}' missing 'module' in registry."
    assert class_name is not None, f"'{model_name}' missing a 'class' in registry."

    # 3. Class imports cleanly
    try:
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, class_name)
    except ImportError as e:
        pytest.fail(
            f"'{model_name}': could not import '{module_path}' (extra installed?): {e}"
        )
    except AttributeError as e:
        pytest.fail(
            f"'{model_name}': class '{class_name}' not found in '{module_path}': {e}"
        )

    # 4. Pipeline interface methods exist
    for method in ("train", "sample", "load_from_ref"):
        assert hasattr(model_class, method), (
            f"'{model_name}' ({class_name}) missing required method '{method}'."
        )

    # 5. Declares ARTIFACT_STATE_FILES
    assert getattr(model_class, "ARTIFACT_STATE_FILES", ()), (
        f"'{model_name}' does not declare non-empty ARTIFACT_STATE_FILES. State will not persist."
    )

    # 6. Integration test exists for model
    test_file = f"tests/test_integration_{model_name}.py"
    assert os.path.isfile(test_file), (
        f"'{model_name}' marked supported but has no integration test at {test_file}."
    )


def test_supported_models_list():
    assert set(ModelRegistry.get_supported_models()) == {"ganblr", "pategan"}
