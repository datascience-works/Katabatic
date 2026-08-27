# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-27

### Added

- Six-dimension evaluation pipeline: fidelity, utility, diversity, privacy, consistency, and stability (`katabatic.pipeline.evaluation_pipeline`).
- Six additional models registered as experimental: `codi`, `ctgan`, `medgan`, `pategan`, `tabddpm`, `tabsyn`.
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

### Fixed

- CTGAN column-type inference treated low-cardinality integer columns as continuous on small datasets, producing continuous synthetic targets that broke TSTR evaluation.
- GANBLR did not persist model state, so trained models could not be reloaded from an artifact reference.

### Security

- `torch` 2.9.0 -> 2.13.0, resolving CVE-2025-2999, CVE-2025-3001, PYSEC-2026-139, PYSEC-2026-2286, and PYSEC-2025-194.
- `transformers` 4.53.2 -> 4.57.0, resolving PYSEC-2025-211 through -216 and -218. Remaining advisories require the v5 major upgrade and are deferred pending GReaT's modernisation.
- `tensorflow` 2.19 -> 2.21.
- Docker container hardened to run as a non-root user.
- Dependency vulnerability scanning (`pip-audit`) and static analysis (`bandit`) added to CI.

### Removed

- Stale per-model `pyproject.toml` and `poetry.lock` files.
- `dev_deps.py`, repo-root `main.py` and `utils.py`, and `katabatic/models/ganblr/kdb.py`.


## [0.1.0a1] - 2026-05-22

First public **alpha** release on TestPyPI / PyPI.

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

[0.1.0a1]: https://github.com/datascience-works/Katabatic/releases/tag/v0.1.0a1
[0.1.0]: https://github.com/datascience-works/Katabatic/releases/tag/v0.1.0
