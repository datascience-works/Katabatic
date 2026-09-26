# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- TVAE-GAN promoted to supported: moved from `katabatic.experimental.models.tvaegan` to `katabatic.models.tvaegan`, with artifact persistence (`load_from_ref`), a `tvaegan` extra, CI integration tests and benchmark scripts for all five datasets. It now uses the shared six-dimension `evaluate()`; its reconstruction loss is available as `evaluate_loss()`.
- TabKDE promoted to supported: `katabatic.experimental.models.tabkde_updated` renamed and moved to `katabatic.models.tabkde`, now subclassing `Model` with artifact persistence (`load_from_ref`) and CI integration tests.

### Fixed
- The privacy dimension failed when a categorical column held integers in the real data and floats in the synthetic data (e.g. `education-num` on adult); values now compare by number, so 13 and 13.0 match.
- TabKDE: numeric class labels were returned as interpolated fractions (e.g. 0.37) by both `train()` and `fit(x, y)`; labels now round-trip exactly.
- TabKDE: `seed` did not reach the GMM radius draw, so seeded draws were not reproducible and a fixed `random_state` returned identical rows on every `sample()` call.

## [1.0.0] - 2026-09-25

The first stable release of Katabatic: a single interface for training, sampling, evaluating, and versioning synthetic tabular data models.

### Highlights

- **15 supported generative models behind one interface.** GAN-based (GANBLR, CTGAN, PATE-GAN), diffusion (TabSyn, FairTabDiffusion), language-model (GReaT, REaLTabFormer), differentially private (MST, PrivTree, PATE-GAN) and statistical baselines (ARF, KDE, Histogram, NaiveBayes, SMOTE, SynthPop). Every model trains with `train(data_dir)`, generates rows with `sample(n_samples)` and reloads a trained model from its artifacts with `load_from_ref()`, and each has an integration test run in CI.
- **Models by name.** `get_model("ctgan")` and `list_supported_models()` find models through `ModelRegistry`. Each model installs as its own extra, e.g. `pip install "katabatic[ganblr,ctgan]"`, and a missing dependency raises an error naming what to install. MST also needs Private-PGM, which installs separately because it isn't on PyPI.
- **Six-dimension evaluation.** `model.evaluate(real_df)` scores fidelity, utility (train on synthetic, test on real), diversity, privacy, consistency and stability, and combines them into a weighted composite. Run a subset with `dimensions=`, or pass your own evaluation as `pipeline=`, as scikit-learn takes `scoring=`. A dimension that fails is reported in `report.errors` rather than scored.
- **End-to-end pipeline with versioned artifacts.** `TrainTestSplitPipeline` splits a CSV, trains a model, generates synthetic data and evaluates it, recording the dataset split, trained model and evaluations in an artifact store so any run can be reloaded or re-evaluated. `LocalArtifactStore` keeps them on disk.
- **Cloud artifact storage (experimental).** `FsspecArtifactStore` keeps artifacts in S3, GCS or Azure through a local cache, so a model trained on one machine can be evaluated on another. A write that would overwrite a newer change from another machine raises `ArtifactConflictError` instead. Install with the `artifacts-s3`, `artifacts-gcs` or `artifacts-azure` extra.
- **Datasets.** Five benchmark datasets ship with the package (adult, car, magic, nursery and shuttle). `DatasetRegistry` profiles any CSV and can check it against a model's requirements, and `preprocess_tabular()` cleans raw CSVs before training.
- **Examples and benchmarks.** Notebooks for the quickstart, evaluation and cloud storage run in CI, and per-model benchmark scripts cover the five datasets.
- **Experimental models** (TabEBM, TabDDPM, GANBLR++, CoDi, MedGAN, TVAE-GAN, GMM and TabKDE) live in `katabatic.experimental.models`, outside the stability guarantee. A model is promoted to supported once it passes the promotion contract, which the test suite enforces.
- **Tooling.** Python 3.11 with Poetry, a Dockerfile, Makefile targets that mirror CI, and pre-commit hooks for Ruff and conventional commits.

### Upgrading from 0.3.1

- Experimental models moved to `katabatic.experimental.models.<name>`.
- The `all` extra is gone: install the extras for the models you use.
- `evaluate()` now runs the six-dimension evaluation on every model. Model-specific scores moved to `evaluate_tstr()` (GANBLR, PATE-GAN), `evaluate_ks()` (ARF) and `evaluate_loss()` (TabSyn, TabDDPM).

## [0.3.1] - 2026-09-12

### Fixed

- `publish-pypi.yml` triggered on both `release: published` and `push: tags: v*`, so publishing a GitHub Release from a fresh tag fired both events and started two runs against the same version; only the first could succeed since PyPI rejects re-uploading an existing version/filename, even from a legitimate rebuild. Removed the redundant tag-push trigger, leaving `release: published` (and manual dispatch) as the only ways to publish.
- `publish-pypi.yml` unconditionally passed `password: ${{ secrets.PYPI_API_TOKEN }}`, which disables PyPI Trusted Publishing even when configured. Reverted to Trusted Publishing (the workflow already carries the required `id-token: write` permission) and added `skip-existing: true` as a safety net against future duplicate-publish attempts.
- Ruff import-order violation in `katabatic/models/great/great_dataset.py`.

## [0.3.0] - 2026-09-12

### Added

- New experimental models: SynthPop (`katabatic.models.synthpop`), GMM, Naive Bayes, SMOTE (numeric-only oversampling), and an updated TabKDE, each with a benchmark evaluation script.
- PATE-GAN and TabSyn promoted to supported: rebuilt/aligned with their original paper and reference implementations, `load_from_ref` implemented, integration tests added, and new `pategan`/`tabsyn` pytest markers registered. The registry now lists four supported models: `ganblr`, `ctgan`, `pategan`, `tabsyn`.
- `smote` PyPI extra (`imbalanced-learn`).
- Cross-platform setup with Windows support, including a CI job that verifies installs across operating systems.
- Data dictionary documenting the five benchmark datasets (`datasets/README.md`).
- `pre-commit` check folded into the local `make ci` run, matching CI.

### Changed

- PyPI classifier moved from Alpha to Beta.
- Poetry 2.4.1 -> 2.4.2; `torch` 2.13.0 -> 2.14.0; `tensorflow-io` 0.31.0 -> 0.37.1; `xgboost` 3.0.2 -> 3.2.0; `ruff` 0.16.4 -> 0.16.6.
- `statsmodels` dropped from the `all` extra in favour of `imbalanced-learn` (for SMOTE).
- Coverage `omit` list updated to reflect current promotions: `pategan` and `tabsyn` now covered; `gmm`, `naivebayes`, and `smote` added as newly-registered experimental models.

### Fixed

- Removed a dangling `katabatic` console-script entry point in `pyproject.toml` left over after the legacy CLI (`init-model`, `register-dataset`, `pin-notebook-kernel`) was removed; installing the package and running `katabatic` would otherwise fail with `ModuleNotFoundError`.

### Removed

- Legacy CLI (`katabatic/cli/`): `init-model`, `register-dataset`, and `pin-notebook-kernel` commands.
- Large tracked data files no longer needed in version control: stale `sample_data/`, `discretized_data/`, `Results/`, `benchmarks/results/` artifacts, `.DS_Store` files, notebook checkpoint output, and outdated setup/pin scripts. `katabatic/datasets/` is now gitignored (aside from its `README.md`).
- TabEBM model, added and removed within this release cycle; not shipped.

## [0.2.0] - 2026-08-28

### Added

- Six-dimension evaluation pipeline: fidelity, utility, diversity, privacy, consistency, and stability (`katabatic.pipeline.evaluation_pipeline`).
- Four additional models registered as experimental: `ctgan`, `pategan`, `tabddpm`, `tabsyn`. (`codi` and `medgan` ship as source but are not in the registry.)
- Model promotion contract, enforced automatically by the `model-contract` CI job: registry entry with matching extra, importable module and class, pipeline interface (`train`, `sample`, `load_from_ref`), non-empty `ARTIFACT_STATE_FILES`, and an integration test.
- Parameterised contract test harness (`tests/test_model_registry.py`) covering every model marked supported.
- CTGAN integration test with a full artifact round-trip: state persistence, reload via `load_from_ref`, and sampling with column-order verification.
- CI jobs: pre-commit, bandit and pip-audit security scanning, coverage gating, per-model integration matrix with path filtering, Docker build and smoke test, and combined coverage reporting across jobs.
- Dockerfile with build-time extras selection (`MODEL_EXTRA`), CPU-only PyTorch, layer-split dependency install, and a non-root runtime user.
- Developer tooling: pre-commit hooks, Makefile targets mirroring CI (`make ci`, `make contract`, `make hooks`), and a generic `make install-model MODEL=x`.
- Repository governance: branch protection on `main` and `development`, CODEOWNERS restricting merges on protected branches, and Dependabot configuration.
- `ModelRegistry.get_model_config()` public accessor.

### Changed

- Migrated packaging to the PEP 621 `[project]` standard; model dependencies moved from core into extras.
- Poetry 1.x -> 2.x.
- GANBLR promoted to supported: declares `ARTIFACT_STATE_FILES`, persists state to the artifact store, and honours the `train_epochs` keyword.
- CTGAN promoted to supported.
- GReaT reverted to experimental pending modernisation to the current model contract (no state persistence).
- GReaT removed from the CI integration matrix while unsupported; its tests remain skipped pending modernisation to the current model contract.

### Fixed

- CTGAN column-type inference treated low-cardinality integer columns as continuous on small datasets, producing continuous synthetic targets that broke TSTR evaluation.
- GANBLR did not persist model state, so trained models could not be reloaded from an artifact reference.
- `katabatic.__version__` is now read from installed package metadata, making `pyproject.toml` the single source of truth.
- Constrained `huggingface-hub` to `<1.0` and `fsspec` to `<=2026.2.0` in the `great` and `all` extras, resolving incompatibilities with `transformers` 4.57 and `datasets` 4.x.
- Removed `tabpfn` from the `all` extra: it requires `huggingface-hub>=1.0`, which `transformers` 4.x rejects. The standalone `tabpfgen` extra is unaffected.

### Security

- `torch` 2.9.0 -> 2.13.0, resolving CVE-2025-2999, CVE-2025-3001, PYSEC-2026-139, PYSEC-2026-2286, and PYSEC-2025-194.
- `transformers` 4.53.2 -> 4.57.0, resolving PYSEC-2025-211 through -216 and -218. Remaining advisories require the v5 major upgrade and are deferred pending GReaT's modernisation.
- `tensorflow` 2.19 -> 2.21.
- Docker container hardened to run as a non-root user.
- Dependency vulnerability scanning (`pip-audit`) and static analysis (`bandit`) added to CI.

### Removed

- Stale per-model `pyproject.toml` and `poetry.lock` files.
- `dev_deps.py`, repo-root `main.py` and `utils.py`, and `katabatic/models/ganblr/kdb.py`.

## [0.1.0] - 2026-05-22

### Added

- PyPI packaging with lean core install (`pandas`, `numpy`, `scikit-learn`) and optional extras (`ganblr`, `great`, `eval`, and experimental model extras).
- `katabatic` CLI: `init-model`, `register-dataset`, `pin-notebook-kernel`.
- Packaged preprocessing API at `katabatic.utils.preprocess` (`preprocess_tabular`, `encode_preprocess`).
- GitHub Actions CI: `poetry check`, `ruff`, fast `pytest`, wheel build, optional integration job for GANBLR.
- Documentation: install matrix, artifact-first quick start, `docs/EXPERIMENTAL_MODELS.md`.
- Integration smoke tests for GANBLR (artifact pipeline) and GReaT (import/registry wiring).
- `ModelRegistry` `supported` flag for officially supported models (`ganblr`, `great`).

### Changed

- Heavy dependencies (TensorFlow, PyTorch, Jupyter, etc.) moved out of the default install into extras or dev dependencies.
- Root `utils.py` is a deprecation shim; import from `katabatic.utils.preprocess` instead.
- README and CONTRIBUTING updated for `datascience-works/Katabatic` URLs and current dev tooling (`ruff`, `poetry install --with dev`).

### Fixed

- Removed invalid `libzero` extra reference that broke `poetry check`.
- Removed `sys.path` hack from GANBLR model module for installed-package compatibility.

### Deprecated

- Repo-root `utils` module (use `katabatic.utils.preprocess`).
- `python main.py` entry point (use `katabatic` CLI).

## [0.1.0a1] - 2026-05-22

First public **alpha** release on TestPyPI / PyPI.

[1.0.0]: https://github.com/datascience-works/Katabatic/releases/tag/v1.0.0
[0.3.1]: https://github.com/datascience-works/Katabatic/releases/tag/v0.3.1
[0.3.0]: https://github.com/datascience-works/Katabatic/releases/tag/v0.3.0
[0.2.0]: https://github.com/datascience-works/Katabatic/releases/tag/v0.2.0
[0.1.0]: https://github.com/datascience-works/Katabatic/releases/tag/v0.1.0
[0.1.0a1]: https://github.com/datascience-works/Katabatic/releases/tag/v0.1.0a1
