# Experimental Models

Katabatic ships multiple generative model implementations. Only a subset is **officially supported**; the rest are experimental. `ModelRegistry.get_supported_models()` in [katabatic/models/registry.py](../katabatic/models/registry.py) is authoritative — the tables below mirror its current contents.

## Supported Models

| Model | Extra | Smoke-tested |
| ------- | ------- | -------------- |
| GANBLR | `pip install katabatic[ganblr]` | Yes (artifact pipeline integration test) |
| CTGAN | `pip install katabatic[ctgan]` | Yes (artifact pipeline integration test) |
| PATE-GAN | `pip install katabatic[pategan]` | Yes (artifact pipeline integration test) |
| TabSyn | `pip install katabatic[tabsyn]` | Yes (artifact pipeline integration test) |
| GReaT | `pip install katabatic[great]` | Yes (artifact pipeline integration test) |
| SMOTE | `pip install katabatic[smote]` | Yes (artifact pipeline integration test) |
| MST | `pip install katabatic[mst]` | Yes (artifact pipeline integration test) |
| PrivTree | `pip install katabatic[privtree]` | Yes (artifact pipeline integration test) |
| ARF | `pip install katabatic[arf]` | Yes (artifact pipeline integration test) |
| SynthPop | `pip install katabatic[synthpop]` | Yes (artifact pipeline integration test; also requires R + CRAN `synthpop` packages) |
| Naive Bayes | `pip install katabatic[naivebayes]` | Yes (artifact pipeline integration test) |
| REaLTabFormer | `pip install katabatic[realtabformer]` | Yes (artifact pipeline integration test) |
| KDE | `pip install katabatic[kde]` | Yes (artifact pipeline integration test) |

These models are listed in `ModelRegistry` with `supported: True`, and `tests/test_model_registry.py::test_supported_models_list` pins this exact set. Use the artifact pipeline documented in [GANBLR_FLOW.md](../GANBLR_FLOW.md) and the README quick start.

## Experimental Models

Present in the codebase, but with no guarantee of **API stability or CI coverage**.
`[tool.coverage.run].omit` in the root `pyproject.toml` is the closest available list of known-experimental models.

### Registered (usable via `ModelRegistry.load_model()`)

| Model | Extra | Notes |
| ------- | ------- | ------- |
| TabDDPM | `tabddpm` | Uses external `tabddpm` package, with a local fallback |

### Not registered (import directly from the module; `ModelRegistry.load_model()` will not find them)

| Model | Extra | Notes |
| ------- | ------- | ------- |
| CoDi | `codi` | See `examples/codi.ipynb` |
| MedGAN | `medgan` | See `examples/medgan.ipynb` |
| TVAE-GAN | *(none)* | |
| GMM | *(none)* | Does not subclass `Model` — implements its own `fit`/`sample`, with no `train`/`evaluate` |
| TabKDE (updated) | *(none)* | Does not subclass `Model` |

Examples under `examples/` and `benchmarks/examples/` are provided for reference and
are not covered by CI.

## Promoting a Model to Supported

See [MODEL_CONTRIBUTIONS.md](../MODEL_CONTRIBUTIONS.md#promoting-a-model-from-experimental-to-supported)
for the full checklist (test harness, promotion contract, CI wiring, dependency extras). In
short: add a PyPI extra and integration test, wire the model into the `changes` job in
[.github/workflows/ci.yml](../.github/workflows/ci.yml), then flip `supported: True` in
`katabatic/models/registry.py` and update this file and the README install matrix.
