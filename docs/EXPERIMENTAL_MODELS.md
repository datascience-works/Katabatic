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
| Histogram | `pip install katabatic[histogram]` | Yes (artifact pipeline integration test) |
| REaLTabFormer | `pip install katabatic[realtabformer]` | Yes (artifact pipeline integration test) |
| KDE | `pip install katabatic[kde]` | Yes (artifact pipeline integration test) |
| FairTabDiffusion | `pip install katabatic[fairtabdiffusion]` | Yes (artifact pipeline integration test) |
| TabKDE | `pip install katabatic[tabkde]` | Yes (artifact pipeline integration test) |
| TVAE-GAN | `pip install katabatic[tvaegan]` | Yes (artifact pipeline integration test) |

These models are listed in `ModelRegistry` with `supported: True`, and `tests/test_model_registry.py::test_supported_models_list` pins this exact set. Use the artifact pipeline documented in [GANBLR_FLOW.md](../GANBLR_FLOW.md) and the README quick start.

## Experimental Models

Experimental models live in `katabatic.experimental.models` and have no guarantee of **API stability or CI coverage**. Everything outside `katabatic.experimental` follows semantic versioning. `tests/test_model_registry.py` checks that a registered model is supported if and only if it lives outside `katabatic/experimental/`.

### Registered (usable via `ModelRegistry.load_model()`)

| Model | Import | Extra | Notes |
| ------- | ------- | ------- | ------- |
| TabEBM | `katabatic.experimental.models.tabebm.models.TabEBMModel` | `tabebm` | Passes the promotion contract, but samples reproduce training rows; see its README |
| TabDDPM | `katabatic.experimental.models.tabddpm.models.Tabddpm` | `tabddpm` | Uses external `tabddpm` package, with a local fallback |
| GANBLR++ | `katabatic.experimental.models.ganblrpp.GANBLRPP` | `ganblr` | GANBLR with numerical columns; no integration test yet |

### Not registered (import directly; `ModelRegistry.load_model()` will not find them)

| Model | Import | Extra | Notes |
| ------- | ------- | ------- | ------- |
| CoDi | `katabatic.experimental.models.codi.CODI` | `codi` | |
| MedGAN | `katabatic.experimental.models.medgan.MEDGAN` | `medgan` | |
| GMM | `katabatic.experimental.models.gmm.GMMModel` | *(none)* | Does not subclass `Model`: implements its own `fit`/`sample`, with no `train`/`evaluate` |

Benchmark scripts for these models are in `benchmarks/examples/<model>/`. They are provided for reference and are not run in CI.

## Promoting a Model to Supported

See [MODEL_CONTRIBUTIONS.md](../MODEL_CONTRIBUTIONS.md#promoting-a-model-from-experimental-to-supported)
for the full checklist (test harness, promotion contract, CI wiring, dependency extras). In
short: add a PyPI extra and integration test, wire the model into the `changes` job in
[.github/workflows/ci.yml](../.github/workflows/ci.yml), move its folder from
`katabatic/experimental/models/` to `katabatic/models/`, then update its registry `module`
path, flip `supported: True` in `katabatic/models/registry.py` and update this file and the
README install matrix.
