"""Model registry for dynamic model loading.

Officially supported models (smoke-tested, PyPI extras): ``ganblr``, ``great``.
Other registered models are experimental; see ``docs/EXPERIMENTAL_MODELS.md``.
"""

from __future__ import annotations

import importlib
from typing import ClassVar

from .base_model import Model


class ModelRegistry:
    """Registry for managing available models and their dependencies."""

    _models: ClassVar[dict[str, dict]] = {
        "ganblr": {
            "module": "katabatic.models.ganblr.models",
            "class": "GANBLR",
            "dependencies": ["tensorflow", "pgmpy", "pyitlib", "tf_keras", "scipy"],
            "extra": "ganblr",
            "supported": True,
            "dataset_requirements": {
                "allowed_tasks": [
                    "binary_classification",
                    "multiclass_classification",
                ],
            },
        },
        "great": {
            "module": "katabatic.models.great.models",
            "class": "GReaT",
            "dependencies": ["transformers", "torch"],
            "extra": "great",
            "supported": False,
        },
        "tabsyn": {
            "module": "katabatic.models.tabsyn.models",
            "class": "Tabsyn",
            "dependencies": [],
            "extra": "tabsyn",
            "supported": False,
        },
        "tabkde": {
            "module": "katabatic.models.tabkde.models",
            "class": "TabKDEModel",
            "dependencies": ["torch", "sklearn", "scipy"],
            "extra": "tabkde",
            "supported": False,
        },
        "tabddpm": {
            "module": "katabatic.models.tabddpm.models",
            "class": "Tabddpm",
            "dependencies": [],
            "extra": "tabddpm",
            "supported": False,
        },
        "pategan": {
            "module": "katabatic.models.pategan.models",
            "class": "PATEGANSynthesizer",
            "dependencies": ["tensorflow", "numpy", "pandas"],
            "extra": "pategan",
            "supported": False,
        },
        "ctgan": {
            "module": "katabatic.models.ctgan.models",
            "class": "CTGANModel",
            "dependencies": ["torch", "sklearn"],
            "extra": "ctgan",
            "supported": False,
        },
    }

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get list of officially supported model names."""
        return [name for name, info in cls._models.items() if info.get("supported")]

    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Return the registry config for a model."""
        if model_name not in cls._models:
            raise KeyError(f"Model '{model_name}' is not registerd.")
        return cls._models[model_name]

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """Return True if the model is officially supported."""
        info = cls._models.get(model_name.lower())
        return bool(info and info.get("supported"))

    @classmethod
    def get_model_info(cls, model_name: str) -> dict | None:
        """Get information about a specific model."""
        return cls._models.get(model_name.lower())

    @classmethod
    def load_model(cls, model_name: str) -> type[Model]:
        """Dynamically load a model class."""
        model_name = model_name.lower()

        if model_name not in cls._models:
            available = ", ".join(cls.get_available_models())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_info = cls._models[model_name]

        missing_deps = []
        for dep in model_info["dependencies"]:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise ImportError(
                f"Missing dependencies for {model_name}: {missing_deps}. "
                f"Install with: pip install katabatic[{model_info['extra']}]"
            )

        try:
            module = importlib.import_module(model_info["module"])
            model_class = getattr(module, model_info["class"])
            return model_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load model {model_name}: {e}")

    @classmethod
    def create_model(cls, model_name: str, *args, **kwargs) -> Model:
        """Create an instance of the specified model."""
        model_class = cls.load_model(model_name)
        return model_class(*args, **kwargs)


def get_model(model_name: str, *args, **kwargs) -> Model:
    """Convenience function to create a model instance."""
    return ModelRegistry.create_model(model_name, *args, **kwargs)


def list_models() -> list[str]:
    """Convenience function to list available models."""
    return ModelRegistry.get_available_models()


def list_supported_models() -> list[str]:
    """Convenience function to list officially supported models."""
    return ModelRegistry.get_supported_models()
