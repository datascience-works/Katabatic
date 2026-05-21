# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/datascience-works/Katabatic/releases/tag/v0.1.0
