# Experimental models

Katabatic ships multiple generative model implementations. Only a subset is **officially supported** for v0.1.0.

## Supported models

| Model | Extra | Smoke-tested |
|-------|-------|--------------|
| GANBLR | `pip install katabatic[ganblr]` | Yes (artifact pipeline integration test) |
| GReaT | `pip install katabatic[great]` | Yes (import + registry wiring) |

These models are listed in `ModelRegistry` with `supported: True`. Use the artifact pipeline documented in [GANBLR_FLOW.md](../GANBLR_FLOW.md) and the README quick start.

## Experimental models

The following are available in the codebase and may be loaded via optional extras, but **API stability and CI coverage are not guaranteed**:

| Model | Extra | Notes |
|-------|-------|-------|
| TabSyn | `tabsyn` | Heavy torch stack |
| TabDDPM | `tabddpm` | Uses external `tabddpm` or local fallback |
| PATE-GAN | `pategan` | TensorFlow |
| CTGAN | `ctgan` | PyTorch |
| CoDi | `codi` | Not in registry; see `examples/codi.ipynb` |
| MedGAN | `medgan` | Not in registry; see `examples/medgan.ipynb` |

Examples under `examples/` for experimental models are best-effort. New contributions start as experimental until a maintainer adds an extra, registry entry (if applicable), and integration smoke coverage.

## Promoting a model to supported

1. Add or verify a PyPI extra in `pyproject.toml`.
2. Register the model in `katabatic/models/registry.py` with `supported: True`.
3. Add a fast integration test (or document why full train is manual-only).
4. Update README install matrix and this file.
