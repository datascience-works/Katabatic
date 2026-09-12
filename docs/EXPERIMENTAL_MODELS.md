# Experimental models

Katabatic ships multiple generative model implementations. Only a subset is **officially supported**; the rest are experimental. `ModelRegistry.get_supported_models()` in [katabatic/models/registry.py](../katabatic/models/registry.py) is the source of truth — the tables below reflect its current contents.

## Supported models

| Model | Extra | Smoke-tested |
|-------|-------|--------------|
| GANBLR | `pip install katabatic[ganblr]` | Yes (artifact pipeline integration test) |
| CTGAN | `pip install katabatic[ctgan]` | Yes (artifact pipeline integration test) |
| PATE-GAN | `pip install katabatic[pategan]` | Yes (artifact pipeline integration test) |

These models are listed in `ModelRegistry` with `supported: True`, and `tests/test_model_registry.py::test_supported_models_list` pins this exact set. Use the artifact pipeline documented in [GANBLR_FLOW.md](../GANBLR_FLOW.md) and the README quick start.

## Experimental models

Available in the codebase and installable via optional extras, but **API stability
and CI coverage are not guaranteed**:

| Model | Extra | Notes |
|-------|-------|-------|
| GReaT | `great` | Has an integration test and CI job, but registered with `supported: False` — not yet promoted |
| AIM | `aim` | Differentially private marginal-based synthesis using Private-PGM; has an artifact pipeline integration test and is registered with `supported: False` |
| TabSyn | `tabsyn` | Heavy torch stack |
| TabDDPM | `tabddpm` | Uses external `tabddpm` or local fallback |
| CoDi | `codi` | Not in registry; see `examples/codi.ipynb` |
| MedGAN | `medgan` | Not in registry; see `examples/medgan.ipynb` |

Models noted "Not in registry" ship as source but can't be loaded through
`ModelRegistry.load_model()` — import them directly from their module instead.

Examples under `examples/` are best-effort. New contributions start as experimental
until a maintainer adds an extra, a registry entry, and integration smoke coverage.

## Promoting a model to supported

1. Add or verify a PyPI extra for the model under `[project.optional-dependencies]` in the root `pyproject.toml` (no per-model `pyproject.toml`/`poetry.lock`), and run `poetry lock`.
2. Add an integration smoke test at `tests/test_integration_<model_name>.py` that exercises the real artifact pipeline (train → sample → evaluate, state persisted via `ARTIFACT_STATE_FILES` and reloaded via `load_from_ref`).
3. Add the model to the `ALL_MODELS` matrix and `paths-filter` block in [.github/workflows/ci.yml](../.github/workflows/ci.yml) so its integration job runs in CI.
4. Register the model in `katabatic/models/registry.py` with `supported: True`, and confirm it passes the promotion contract in `tests/test_model_registry.py::test_model_promotion_contract` (also update `test_supported_models_list`'s expected set).
5. Update the README install matrix and this file.

See [MODEL_CONTRIBUTIONS.md](../MODEL_CONTRIBUTIONS.md#promoting-a-model-from-experimental-to-supported) for the full promotion checklist.
