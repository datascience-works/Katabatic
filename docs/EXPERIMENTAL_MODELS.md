# Experimental models

Katabatic ships multiple generative model implementations. Only a subset is
**officially supported**; the rest are experimental.

## Supported models

| Model | Extra | Covered by |
|-------|-------|------------|
| GANBLR | `pip install katabatic[ganblr]` | `tests/test_integration_ganblr.py` (artifact pipeline round-trip) |
| CTGAN | `pip install katabatic[ctgan]` | `tests/test_integration_ctgan.py` (artifact pipeline round-trip) |
| PATE-GAN | `pip install katabatic[pategan]` | `tests/test_integration_pategan.py` (import, registry, artifact reload) |

These are the models listed in `ModelRegistry` with `supported: True`, enforced by
the `model-contract` CI job via `tests/test_model_registry.py`. Use the artifact
pipeline documented in [GANBLR_FLOW.md](../GANBLR_FLOW.md) and the README quick start.

## Experimental models

Available in the codebase and installable via optional extras, but **API stability
and CI coverage are not guaranteed**:

| Model | Extra | In registry | Notes |
|-------|-------|-------------|-------|
| GReaT | `great` | Yes | Reverted to experimental in 0.2.0 pending modernisation to the current model contract (no state persistence). |
| TabSyn | `tabsyn` | Yes | Heavy torch stack |
| TabDDPM | `tabddpm` | Yes | Requires the external `tabddpm` package, which the extra does not install |
| CoDi | `codi` | No | See `examples/codi.ipynb` |
| MedGAN | `medgan` | No | See `examples/medgan.ipynb` |
| SMOTE | `smote` | No | Oversampling baseline, not a generative model |
| GMM | — | No | No extra; uses core scikit-learn |
| Naive Bayes | — | No | No extra; uses core scikit-learn |
| TabKDE (updated) | — | No | No extra |

Models marked "In registry: No" ship as source but cannot be loaded through
`ModelRegistry.load_model()`; import them directly from their module.

Examples under `examples/` for experimental models are best-effort. New
contributions start as experimental until a maintainer adds an extra, a registry
entry, and integration smoke coverage.

## Promoting a model to supported

1. Add or verify a PyPI extra in `pyproject.toml` (the extra name must equal the
   model name).
2. Register the model in `katabatic/models/registry.py` with `supported: True`.
3. Declare a non-empty tuple `ARTIFACT_STATE_FILES` and implement `load_from_ref`
   so trained state survives a pipeline round-trip.
4. Add a fast integration test at `tests/test_integration_<model>.py`.
5. Add the model to the CI integration matrix in `.github/workflows/ci.yml`.
6. Update the README install matrix, this file, and `CHANGELOG.md`.
