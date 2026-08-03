import pytest
import importlib
from katabatic.models.registry import ModelRegistry

def get_supported_models():
    """
    Directly accesses ModelRegistry._models to extract supported models.
    """
    supported_models = []
    
    # Directly access the _models dictionary shown in your screenshot
    registry_dict = ModelRegistry._models
    
    for model_name, config in registry_dict.items():
        if config.get("supported") is True:
            supported_models.append((model_name, config))
            
    if not supported_models:
        raise ValueError("CRITICAL ERROR: Accessed ModelRegistry._models but found 0 models with 'supported: True'.")
        
    return supported_models

# Fetch the models before running the tests so Pytest knows what to process
MODELS_TO_TEST = get_supported_models()

@pytest.mark.parametrize("model_name, config", MODELS_TO_TEST)
def test_model_promotion_contract(model_name, config):
    """
    Contract test: Validates required registry entries and model methods for promoted models.
    """
    # 1. Verify 'extra' exists in the registry dictionary
    assert "extra" in config, f"Contract Violation: '{model_name}' missing 'extra' in registry."
    assert config["extra"] is not None, f"Contract Violation: '{model_name}' 'extra' is null."

    # 2. Extract module and class paths to inspect the actual model code
    module_path = config.get("module")
    class_name = config.get("class")

    assert module_path is not None, f"Registry Error: '{model_name}' is missing a 'module' path."
    assert class_name is not None, f"Registry Error: '{model_name}' is missing a 'class' name."

    # 3. Dynamically import the model class
    try:
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, class_name)
    except ImportError as e:
        pytest.fail(f"Import Error: Could not load module '{module_path}' for '{model_name}'. Exception: {e}")
    except AttributeError as e:
        pytest.fail(f"Import Error: Could not find class '{class_name}' inside '{module_path}'. Exception: {e}")

    # 4. Verify 'load_from_ref' exists on the class
    assert hasattr(model_class, "load_from_ref"), \
        f"Contract Violation: The class '{class_name}' for '{model_name}' is missing the 'load_from_ref' method."

    # 5. Verify 'smoke_test' exists
    assert hasattr(model_class, "smoke_test") or "smoke_test" in config, \
        f"Contract Violation: '{model_name}' is missing a 'smoke_test' implementation."